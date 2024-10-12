# pylint: disable=invalid-name
"""Module with CycleGAN class."""
import torch
from torch import nn
from .basemodel import BaseModel
from .networks import Generator, Discriminator, CycleGANLoss

class CycleGAN(BaseModel):
    """
    CycleGAN model for image-to-image translation.

    Args:
    - input_nc: Number of input channels. Default is 3.
    - output_nc: Number of output channels. Default is 3.
    - n_residual_blocks: Number of residual blocks in generators. Default is 9.
    - n_features: Number of features in generators and discriminators. Default is 64.
    - n_downsampling: Number of downsampling layers in generators. Default is 2.
    - cycle_loss_weight: Weight for cycle-consistency loss. Default is 10.
    - id_loss_weight: Weight for identity loss. Default is 5.
    - lr: Learning rate. Default is 0.0002.
    - beta1: Beta1 for Adam optimizer. Default is 0.5.
    - beta2: Beta2 for Adam optimizer. Default is 0.999.
    - device: 'cuda' or 'cpu'. Default is 'cpu'.
    """
    def __init__(self, input_nc=3, output_nc=3,
                 n_residual_blocks=9, n_features=64, n_downsampling=2,
                 cycle_loss_weight=10, id_loss_weight=5,
                 lr=0.0002, beta1=0.5, beta2=0.999, device='cpu'):
        super().__init__(device)

        # Initialize generators and discriminators
        gen_params = {
            'input_nc': input_nc,
            'output_nc': output_nc,
            'n_residual_blocks': n_residual_blocks,
            'n_features': n_features,
            'n_downsampling': n_downsampling
        }
        self.gen_AtoB = Generator(**gen_params).to(self.device)
        self.gen_BtoA = Generator(**gen_params).to(self.device)
        self.dis_A = Discriminator(input_nc, n_features=n_features).to(self.device)
        self.dis_B = Discriminator(input_nc, n_features=n_features).to(self.device)

        # Define losses
        self.adversarial_loss = CycleGANLoss().to(self.device)
        self.cycle_loss = nn.L1Loss().to(self.device)
        self.identity_loss = nn.L1Loss().to(self.device)

        # Setup optimizers using the BaseModel's helper function
        self.optimizer_G = self.setup_optimizers(
            list(self.gen_AtoB.parameters()) + list(self.gen_BtoA.parameters()), lr, beta1, beta2
        )
        self.optimizer_D_A = self.setup_optimizers(self.dis_A.parameters(), lr, beta1, beta2)
        self.optimizer_D_B = self.setup_optimizers(self.dis_B.parameters(), lr, beta1, beta2)

        self.device = device
        self.cycle_loss_weight = cycle_loss_weight
        self.id_loss_weight = id_loss_weight

    def __str__(self):
        """String representation of the CycleGAN model."""
        return (
            f'CycleGAN Model\n'
            f'Generators:\n'
            f'  A to B: {self.gen_AtoB}\n'
            f'  B to A: {self.gen_BtoA}\n'
            f'Discriminators:\n'
            f'  A: {self.dis_A}\n'
            f'  B: {self.dis_B}\n'
            f'Losses:\n'
            f'  Adversarial: {self.adversarial_loss}\n'
            f'  Cycle: {self.cycle_loss}\n'
            f'  Identity: {self.identity_loss}\n'
        )

    def eval(self):
        """
        Set the CycleGAN model and its submodules to evaluation mode.
        """
        self.gen_AtoB.eval()
        self.gen_BtoA.eval()
        self.dis_A.eval()
        self.dis_B.eval()

    def train(self):
        """
        Set the CycleGAN model and its submodules to training mode.
        """
        self.gen_AtoB.train()
        self.gen_BtoA.train()
        self.dis_A.train()
        self.dis_B.train()

    def state_dict(self):
        """
        Get the model state dictionary.
        """
        return {
            'gen_AtoB': self.gen_AtoB.state_dict(),
            'gen_BtoA': self.gen_BtoA.state_dict(),
            'dis_A': self.dis_A.state_dict(),
            'dis_B': self.dis_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict(),
        }


    def forward(self, real_A, real_B): # pylint: disable=arguments-differ
        """
        Forward pass for both generators.
        """
        fake_B = self.gen_AtoB(real_A)
        fake_A = self.gen_BtoA(real_B)

        return fake_B, fake_A

    def compute_loss(self, real_A, real_B): # pylint: disable=arguments-differ
        """
        Computes the total loss for generators and discriminators
        using CycleGANLoss for adversarial loss.
        """
        fake_B, fake_A = self.forward(real_A, real_B)

        # Identity loss
        loss_identity_A = self.identity_loss(self.gen_BtoA(real_A), real_A)
        loss_identity_B = self.identity_loss(self.gen_AtoB(real_B), real_B)

        # GAN loss using CycleGANLoss
        loss_G_AtoB = self.adversarial_loss(self.dis_B(fake_B), target_is_real=True)
        loss_G_BtoA = self.adversarial_loss(self.dis_A(fake_A), target_is_real=True)

        # Cycle-consistency loss
        loss_cycle_A = self.cycle_loss(self.gen_BtoA(fake_B), real_A)
        loss_cycle_B = self.cycle_loss(self.gen_AtoB(fake_A), real_B)

        # Total generator loss
        loss_G_ad = loss_G_AtoB + loss_G_BtoA
        loss_G_cycle = loss_cycle_A + loss_cycle_B
        loss_G_id = loss_identity_A + loss_identity_B
        loss_G = loss_G_ad + self.cycle_loss_weight * loss_G_cycle + self.id_loss_weight * loss_G_id

        # Discriminator A loss (real vs fake)
        loss_real_A = self.adversarial_loss(self.dis_A(real_A), target_is_real=True)
        loss_fake_A = self.adversarial_loss(self.dis_A(fake_A.detach()), target_is_real=False)
        loss_D_A = (loss_real_A + loss_fake_A) * 0.5

        # Discriminator B loss (real vs fake)
        loss_real_B = self.adversarial_loss(self.dis_B(real_B), target_is_real=True)
        loss_fake_B = self.adversarial_loss(self.dis_B(fake_B.detach()), target_is_real=False)
        loss_D_B = (loss_real_B + loss_fake_B) * 0.5

        return loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id

    def optimize(self, real_A, real_B): # pylint: disable=arguments-differ
        """
        Perform one optimization step for the generators and discriminators.
        """
        loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id = self.compute_loss(real_A, real_B)

        # Optimize Generators
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        # Optimize Discriminator A
        self.optimizer_D_A.zero_grad()
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # Optimize Discriminator B
        self.optimizer_D_B.zero_grad()
        loss_D_B.backward()
        self.optimizer_D_B.step()

        return loss_G.item(), loss_D_A.item(), loss_D_B.item(), loss_G_ad.item(), loss_G_cycle.item(), loss_G_id.item()

    def save_model(self, path='cycle_gan_model.pth'):
        """
        Save the current model state.

        Args:
        - path: Path to save the model. Default is 'cycle_gan_model.pth'.
        """
        torch.save({
            'gen_AtoB': self.gen_AtoB.state_dict(),
            'gen_BtoA': self.gen_BtoA.state_dict(),
            'dis_A': self.dis_A.state_dict(),
            'dis_B': self.dis_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict(),
        }, path)

    def load_model(self, path):
        """
        Load a saved model state.

        Args:
        - path: Path to the saved model.
        """
        checkpoint = torch.load(path, weights_only=True)
        self.gen_AtoB.load_state_dict(checkpoint['gen_AtoB'])
        self.gen_BtoA.load_state_dict(checkpoint['gen_BtoA'])
        self.dis_A.load_state_dict(checkpoint['dis_A'])
        self.dis_B.load_state_dict(checkpoint['dis_B'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
