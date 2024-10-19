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

        cls.out_folder = Path(__file__).resolve().parent.parent / 'no_sync/test_model_skip'
        cls.out_folder.mkdir(parents=True, exist_ok=True)

        cls.hyperparameters = {
            "img_height" : 256,
            "img_width" : 256,

            "batch_size" : 16,
            "n_features" : 32, #64
            "n_residual_blocks": 2, #9
            "n_downsampling": 2, #2

            "norm_type": "instance", #"instance"

            "use_replay_buffer": True, #False
            "replay_buffer_size": 50, #50

            "add_skip": True, #False
            "vanilla_loss": False, #True

            "cycle_loss_weight": 10, #10
            "id_loss_weight": 5, #5
            "plp_loss_weight": 1, #5

            "plp_step": 16, #0
            "plp_beta":0.99, #0.99

            "num_epochs" : 100,
            "device" : torch.device("cuda" if (torch.cuda.is_available() and cls.use_cuda) else "cpu"),

            "lr" : 0.0002, #0.0002
            "beta1" : 0.5,  #0.5
            "beta2" : 0.999, #0.999

            "channels" : 3, #3
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
            add_skip=cls.hyperparameters["add_skip"],
            use_replay_buffer=cls.hyperparameters["use_replay_buffer"],
            replay_buffer_size=cls.hyperparameters["replay_buffer_size"],
            vanilla_loss=cls.hyperparameters["vanilla_loss"],
            cycle_loss_weight=cls.hyperparameters["cycle_loss_weight"],
            id_loss_weight=cls.hyperparameters["id_loss_weight"],
            plp_loss_weight=cls.hyperparameters["plp_loss_weight"],
            plp_step=cls.hyperparameters["plp_step"],
            plp_beta=cls.hyperparameters["plp_beta"],
            lr=cls.hyperparameters["lr"],
            beta1=cls.hyperparameters["beta1"],
            beta2=cls.hyperparameters["beta2"],
        )


        folder = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'nexet'
        train_A_csv = folder / 'input_A_train_filtered.csv'
        test_A_csv = folder / 'input_A_test_filtered.csv'
        train_B_csv = folder / 'input_B_train_filtered.csv'
        test_B_csv = folder / 'input_B_test_filtered.csv'

        transformation = transforms.Compose([
            transforms.Resize(int(cls.hyperparameters["img_height"] * 1.12), transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop((cls.hyperparameters["img_height"], cls.hyperparameters["img_width"])),
            transforms.RandomHorizontalFlip(),
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

        n_train = min(len(cls.train_A.dataset), len(cls.train_B.dataset))
        cls.train_A.dataset.set_len(n_train)
        cls.train_B.dataset.set_len(n_train)
        print(f"Number of training samples: {n_train}")

        n_test = min(len(cls.test_A.dataset), len(cls.test_B.dataset))
        cls.test_A.dataset.set_len(n_test)
        cls.test_B.dataset.set_len(n_test)
        print(f"Number of test samples: {n_test}")

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
        fake_B, fake_A = self.cycle_gan.forward(real_A, real_B)

        if self.print_memory:
            utils.print_gpu_memory_usage("Memory usage after model forward call", short_msg=True)

        self.assertEqual(real_A.shape, fake_B.shape, 'real_A.shape != fake_B.shape')
        self.assertEqual(real_B.shape, fake_A.shape, 'real_B.shape != fake_A.shape')

        real_A = None
        real_B = None
        fake_B = None
        fake_A = None
        torch.cuda.empty_cache()


    def test_few_epochs(self):
        """Test running few epochs."""
        print("Testing running few epochs")

        utils.remove_all_files(self.out_folder)
        train_losses_G, train_losses_D_A, train_losses_D_B = [], [], []
        train_losses_G_ad, train_losses_G_cycle, train_losses_G_id, train_losses_G_plp = [], [], [], []

        for epoch in range(10):
            loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id, loss_G_plp = utils.train_one_epoch(
                epoch=epoch,
                model=self.cycle_gan,
                train_A=self.train_A,
                train_B=self.train_B,
                device=self.hyperparameters["device"],
                n_samples=None,
                plp_step=self.hyperparameters["plp_step"],
            )

            train_losses_G.append(loss_G)
            train_losses_D_A.append(loss_D_A)
            train_losses_D_B.append(loss_D_B)
            train_losses_G_ad.append(loss_G_ad)
            train_losses_G_cycle.append(loss_G_cycle)
            train_losses_G_id.append(loss_G_id)
            train_losses_G_plp.append(loss_G_plp)

            # Save the average losses to a file
            utils.save_losses(
                train_losses_G, train_losses_D_A, train_losses_D_B,
                train_losses_G_ad, train_losses_G_cycle,
                train_losses_G_id, train_losses_G_plp,
                filename=self.out_folder / 'train_losses.txt')

            if epoch % self.hyperparameters["checkpoint_interval"] == 0:
                self.cycle_gan.save_model(self.out_folder / f'cycle_gan_epoch_{epoch}.pth')

            if self.run_wnadb:
                wandb.log({
                    'G_loss/train': loss_G,
                    'D_A_loss/train': loss_D_A,
                    'D_B_loss/train': loss_D_B,
                    'G_loss_ad/train': loss_G_ad,
                    'G_loss_cycle/train': loss_G_cycle,
                    'G_loss_id/train': loss_G_id,
                    'G_loss_plp/train': loss_G_plp,
                })

            real_A = next(iter(self.test_A))
            real_B = next(iter(self.test_B))

            n_images = 4
            if self.use_cuda:
                real_A = real_A.cuda()
                real_B = real_B.cuda()

            imgs_A, imgs_B = self.cycle_gan.generate_samples(real_A, real_B, n_images=n_images)

            utils.show_img(imgs_A, title=f'Epoch {epoch} - A Images',
                        figsize = (20, 16), change_scale=True, nrow=n_images,
                        labels=['Real', 'Fake', 'Recovered', 'Identity'])
            plt.savefig(self.out_folder / f'imgs_{epoch}_A.png')

            utils.show_img(imgs_B, title=f'Epoch {epoch} - B Images',
                        figsize = (20, 16), change_scale=True, nrow=n_images,
                        labels=['Real', 'Fake', 'Recovered', 'Identity'])
            plt.savefig(self.out_folder / f'imgs_{epoch}_B.png')

    def test_reading_model(self):
        """Test reading pth files."""
        print("Testing reading model")

        n = self.hyperparameters["checkpoint_interval"]
        self.cycle_gan.load_model(self.out_folder / f'cycle_gan_epoch_{n}.pth')

        real_A = next(iter(self.test_A))
        real_B = next(iter(self.test_B))

        n_images = 4
        if self.use_cuda:
            real_A = real_A.cuda()
            real_B = real_B.cuda()

        imgs_A, imgs_B = self.cycle_gan.generate_samples(real_A, real_B, n_images=n_images)

        utils.show_img(imgs_A, title='A Images',
                       figsize = (20, 16), change_scale=True, nrow=n_images,
                       labels=['Real', 'Fake', 'Recovered', 'Identity'])
        test_file = self.out_folder / 'A_imgs.png'
        plt.savefig(test_file)
        self.assertTrue(test_file.exists(), f"File {test_file.name} does not exist")

        utils.show_img(imgs_B, title='B Images',
                       figsize = (20, 16), change_scale=True, nrow=n_images,
                       labels=['Real', 'Fake', 'Recovered', 'Identity'])
        test_file = self.out_folder / 'B_imgs.png'
        plt.savefig(test_file)
        self.assertTrue(test_file.exists(), f"File {test_file.name} does not exist")

if __name__ == '__main__':
    # Create a test suite with the desired order
    suite = unittest.TestSuite()
    # suite.addTest(TestCycleGAN('test_shapes'))
    suite.addTest(TestCycleGAN('test_few_epochs'))
    # suite.addTest(TestCycleGAN('test_reading_model'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # unittest.main()
