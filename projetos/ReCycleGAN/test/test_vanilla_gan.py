"""Test running vanilla GAN."""

import unittest
import sys
from pathlib import Path
import torch
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from models import CycleGAN  # pylint: disable=all
from metrics import FID, LPIPS  # pylint: disable=all
from utils.data_loader import get_img_dataloader  # pylint: disable=all
from utils import utils # pylint: disable=all

class TestFID(unittest.TestCase):
    def setUp(self):
        self.use_cuda = True
        self.run_wnadb = False
        self.print_memory = True

        self.hyperparameters = {
            "batch_size" : 16,
            "n_features" : 64, #64
            "n_residual_blocks": 1, #9
            "n_downsampling": 1, #2

            "cycle_loss_weight":10,
            "id_loss_weight":5,

            "num_epochs" : 100,
            "device" : torch.device("cuda" if (torch.cuda.is_available() and self.use_cuda) else "cpu"),

            "lr" : 0.0002,
            "beta1" : 0.5,
            "beta2" : 0.999,
            # "n_cpu" : 8,

            "img_size" : 256,
            "channels" : 3,
            # "sample_interval" : 100,
            "checkpoint_interval" : 10,
        }
        self.use_cuda = self.hyperparameters["device"] == torch.device("cuda")
        # cycle_loss_weight=10, id_loss_weight=5,
        # lr=0.0002, beta1=0.5, beta2=0.999, device='cpu'
        print(f'Using device: "{self.hyperparameters["device"]}"')

        if self.use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.print_memory = self.print_memory and self.use_cuda

        folder = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'nexet'
        train_A_csv = folder / 'input_A_train.csv'
        test_A_csv = folder / 'input_A_test.csv'
        train_B_csv = folder / 'input_B_train.csv'
        test_B_csv = folder / 'input_B_test.csv'

        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if self.print_memory:
            utils.print_gpu_memory_usage("Initital memory usage", short_msg=True)

        batch_size = self.hyperparameters["batch_size"]
        self.train_A = get_img_dataloader(csv_file=train_A_csv, batch_size=batch_size, transformation=transformation)
        self.test_A = get_img_dataloader(csv_file=test_A_csv, batch_size=batch_size, transformation=transformation)
        self.train_B = get_img_dataloader(csv_file=train_B_csv, batch_size=batch_size, transformation=transformation)
        self.test_B = get_img_dataloader(csv_file=test_B_csv, batch_size=batch_size, transformation=transformation)

        if self.run_wnadb:
            # from wandb_utils import init_wandb, log_hyperparameters
            # init_wandb()
            # log_hyperparameters(self.hyperparameters)

            import wandb
            wandb.init(project="cyclegan", config=self.hyperparameters)

    def test_shapes(self):
        """Test data shapes."""
        if self.print_memory:
            utils.print_gpu_memory_usage("Memory usage before loading images", short_msg=True)

        cycle_gan = CycleGAN(
            input_nc=self.hyperparameters["channels"],
            output_nc=self.hyperparameters["channels"],
            device=self.hyperparameters["device"],
            n_features=self.hyperparameters["n_features"],
            n_residual_blocks=self.hyperparameters["n_residual_blocks"],
            n_downsampling=self.hyperparameters["n_downsampling"]
        )

        real_A = next(iter(self.train_A))
        real_B = next(iter(self.train_B))

        if self.use_cuda:
            real_A = real_A.cuda()
            real_B = real_B.cuda()
        if self.print_memory:
            utils.print_gpu_memory_usage("Memory usage after loading images", short_msg=True)

        fake_B, fake_A, recovered_A, recovered_B = cycle_gan.forward(real_A, real_B)

        if self.print_memory:
            utils.print_gpu_memory_usage("Memory usage after model forward call", short_msg=True)

        self.assertEqual(real_A.shape, fake_B.shape, 'real_A.shape != fake_B.shape')
        self.assertEqual(real_B.shape, fake_A.shape, 'real_B.shape != fake_A.shape')
        self.assertEqual(real_A.shape, recovered_A.shape, 'real_A.shape != recovered_A.shape')
        self.assertEqual(real_B.shape, recovered_B.shape, 'real_B.shape != recovered_B.shape')

    # def test_few_epcohs(self):
    #     """Test running 5 epochs."""
    #     cycle_gan = CycleGAN(
    #         input_nc=self.hyperparameters["channels"],
    #         output_nc=self.hyperparameters["channels"],
    #         device=self.hyperparameters["device"])

    #     for epoch in range(5):
    #         cycle_gan.train(self.train_A, self.train_B, self.hyperparameters, epoch)


    #     train_losses_G, train_losses_D_A, train_losses_D_B = [], [], []

    #     for epoch in range(5):
    #         loss_G, loss_D_A, loss_D_B = utils.train_one_epoch(
    #             epoch=epoch,
    #             model=cycle_gan,
    #             train_A=self.train_A,
    #             train_B=self.train_B,
    #             device=self.hyperparameters["device"])

    #         avg_loss_G   = loss_G / len(self.train_A)
    #         avg_loss_D_A = loss_D_A / len(self.train_A)
    #         avg_loss_D_B = loss_D_B / len(self.train_B)

    #         train_losses_G.append(avg_loss_G)
    #         train_losses_D_A.append(avg_loss_D_A)
    #         train_losses_D_B.append(avg_loss_D_B)

    #         # Save the average losses to a file
    #         utils.save_losses(train_losses_G, train_losses_D_A, train_losses_D_B)

    #         if epoch % self.hyperparameters["checkpoint_interval"] == 0:
    #             utils.save_model(epoch, cycle_gan, local_path=f'cycle_gan_epoch_{epoch}.pth', wandb_log=True)

    #         if self.run_wnadb:
    #             wandb.log({
    #                 'G_loss/train': avg_loss_G,
    #                 'D_A_loss/train': avg_loss_D_A,
    #                 'D_B_loss/train': avg_loss_D_B,
    #             })

if __name__ == '__main__':
    unittest.main()
