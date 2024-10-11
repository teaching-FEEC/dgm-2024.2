"""Test running vanilla GAN."""

import unittest
import sys
from pathlib import Path
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from models import CycleGAN  # pylint: disable=all
from metrics import FID, LPIPS  # pylint: disable=all
from utils.data_loader import get_img_dataloader  # pylint: disable=all
from utils import utils # pylint: disable=all

class TestCycleGAN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.use_cuda = True
        cls.run_wnadb = False
        cls.print_memory = True

        cls.out_folder = Path(__file__).resolve().parent.parent / 'no_sync/test_model'
        cls.out_folder.mkdir(parents=True, exist_ok=True)
        # utils.remove_all_files(cls.out_folder)

        cls.hyperparameters = {
            "batch_size" : 32,
            "n_features" : 32, #64
            "n_residual_blocks": 2, #9
            "n_downsampling": 2, #2

            "cycle_loss_weight":10, #10
            "id_loss_weight":5, #5

            "num_epochs" : 100,
            "device" : torch.device("cuda" if (torch.cuda.is_available() and cls.use_cuda) else "cpu"),

            "lr" : 0.0002, #0.0002
            "beta1" : 0.5,  #0.5
            "beta2" : 0.999, #0.999
            # "n_cpu" : 8,

            # "img_size" : 256,
            "channels" : 3,
            # "sample_interval" : 100,
            "checkpoint_interval" : 2,
        }

        cls.use_cuda = cls.hyperparameters["device"] == torch.device("cuda")
        print(f'Using device: "{cls.hyperparameters["device"]}"')

        if cls.use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        cls.print_memory = cls.print_memory and cls.use_cuda

        cls.cycle_gan = CycleGAN(
            input_nc=cls.hyperparameters["channels"],
            output_nc=cls.hyperparameters["channels"],
            device=cls.hyperparameters["device"],
            n_features=cls.hyperparameters["n_features"],
            n_residual_blocks=cls.hyperparameters["n_residual_blocks"],
            n_downsampling=cls.hyperparameters["n_downsampling"],
            cycle_loss_weight=cls.hyperparameters["cycle_loss_weight"],
            id_loss_weight=cls.hyperparameters["id_loss_weight"],
            lr=cls.hyperparameters["lr"],
            beta1=cls.hyperparameters["beta1"],
            beta2=cls.hyperparameters["beta2"],
        )


        folder = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'nexet'
        train_A_csv = folder / 'input_A_train.csv'
        test_A_csv = folder / 'input_A_test.csv'
        train_B_csv = folder / 'input_B_train.csv'
        test_B_csv = folder / 'input_B_test.csv'

        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if cls.print_memory:
            utils.print_gpu_memory_usage("Initital memory usage", short_msg=True)

        batch_size = cls.hyperparameters["batch_size"]
        cls.train_A = get_img_dataloader(csv_file=train_A_csv, batch_size=batch_size, transformation=transformation)
        cls.test_A = get_img_dataloader(csv_file=test_A_csv, batch_size=batch_size, transformation=transformation)
        cls.train_B = get_img_dataloader(csv_file=train_B_csv, batch_size=batch_size, transformation=transformation)
        cls.test_B = get_img_dataloader(csv_file=test_B_csv, batch_size=batch_size, transformation=transformation)

        if cls.run_wnadb:
            # from wandb_utils import init_wandb, log_hyperparameters
            # init_wandb()
            # log_hyperparameters(self.hyperparameters)

            import wandb
            wandb.init(project="cyclegan", config=cls.hyperparameters)

    def test_shapes(self):
        """Test data shapes."""
        print('Number of parameters:')
        print(f'  Generator A to B: {utils.count_parameters(self.cycle_gan.gen_AtoB):,}')
        print(f'  Generator B to A: {utils.count_parameters(self.cycle_gan.gen_BtoA):,}')
        print(f'  Discriminator A:  {utils.count_parameters(self.cycle_gan.dis_A):,}')
        print(f'  Discriminator B:  {utils.count_parameters(self.cycle_gan.dis_B):,}')

        if self.print_memory:
            utils.print_gpu_memory_usage("Memory usage before loading images", short_msg=True)

        real_A = next(iter(self.train_A))
        real_B = next(iter(self.train_B))

        if self.use_cuda:
            real_A = real_A.cuda()
            real_B = real_B.cuda()
        if self.print_memory:
            utils.print_gpu_memory_usage("Memory usage after loading images", short_msg=True)

        self.cycle_gan.eval()
        fake_B, fake_A, recovered_A, recovered_B = self.cycle_gan.forward(real_A, real_B)

        if self.print_memory:
            utils.print_gpu_memory_usage("Memory usage after model forward call", short_msg=True)

        self.assertEqual(real_A.shape, fake_B.shape, 'real_A.shape != fake_B.shape')
        self.assertEqual(real_B.shape, fake_A.shape, 'real_B.shape != fake_A.shape')
        self.assertEqual(real_A.shape, recovered_A.shape, 'real_A.shape != recovered_A.shape')
        self.assertEqual(real_B.shape, recovered_B.shape, 'real_B.shape != recovered_B.shape')

        real_A = None
        real_B = None
        fake_B = None
        fake_A = None
        recovered_A = None
        recovered_B = None
        torch.cuda.empty_cache()


    def test_few_epochs(self):
        """Test running few epochs."""
        print("Testing running few epochs")

        train_losses_G, train_losses_D_A, train_losses_D_B = [], [], []

        for epoch in range(self.hyperparameters["checkpoint_interval"]+1):
            loss_G, loss_D_A, loss_D_B = utils.train_one_epoch(
                epoch=epoch,
                model=self.cycle_gan,
                train_A=self.train_A,
                train_B=self.train_B,
                device=self.hyperparameters["device"])

            avg_loss_G   = loss_G / len(self.train_A)
            avg_loss_D_A = loss_D_A / len(self.train_A)
            avg_loss_D_B = loss_D_B / len(self.train_B)

            train_losses_G.append(avg_loss_G)
            train_losses_D_A.append(avg_loss_D_A)
            train_losses_D_B.append(avg_loss_D_B)

            # Save the average losses to a file
            utils.save_losses(train_losses_G, train_losses_D_A, train_losses_D_B, filename=self.out_folder / 'train_losses.txt')

            if epoch % self.hyperparameters["checkpoint_interval"] == 0:
                # self.cycle_gan.save_model(self.out_folder / f'cycle_gan_epoch_{epoch}.pth')
                utils.save_model(
                    self.cycle_gan,
                    local_path=self.out_folder / f'cycle_gan_epoch_{epoch}.pth',
                    wandb_log=self.run_wnadb)

            if self.run_wnadb:
                wandb.log({
                    'G_loss/train': avg_loss_G,
                    'D_A_loss/train': avg_loss_D_A,
                    'D_B_loss/train': avg_loss_D_B,
                })

    def test_reading_model(self):
        """Test reading pth files."""
        print("Testing reading model")

        n = self.hyperparameters["checkpoint_interval"]
        self.cycle_gan.load_model(self.out_folder / f'cycle_gan_epoch_{n}.pth')

        real_A = next(iter(self.train_A))
        real_B = next(iter(self.train_B))

        n_images = 4
        if self.use_cuda:
            real_A = real_A[:n_images].cuda()
            real_B = real_B[:n_images].cuda()

        self.cycle_gan.eval()
        fake_B, fake_A, recovered_A, recovered_B = self.cycle_gan.forward(real_A, real_B)

        utils.show_img(
            torch.vstack([real_A, fake_B, recovered_A]),
            title='A Images', figsize = (20, 12), change_scale=True, nrow=n_images)
        test_file = self.out_folder / 'A_imgs.png'
        plt.savefig(test_file)
        self.assertTrue(test_file.exists(), f"File {test_file.name} does not exist")

        utils.show_img(torch.vstack([real_B, fake_A, recovered_B]),
                       title='B Images', figsize = (20, 12), change_scale=True, nrow=n_images)
        test_file = self.out_folder / 'B_imgs.png'
        plt.savefig(test_file)
        self.assertTrue(test_file.exists(), f"File {test_file.name} does not exist")

if __name__ == '__main__':
    # Create a test suite with the desired order
    suite = unittest.TestSuite()
    # suite.addTest(TestCycleGAN('test_shapes'))
    # suite.addTest(TestCycleGAN('test_few_epochs'))
    suite.addTest(TestCycleGAN('test_reading_model'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # unittest.main()
