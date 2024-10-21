# pylint: disable=invalid-name
"""Module with CycleGAN class."""
import gc
import torch
from torch import nn

from src.utils import Loss
from .basemodel import BaseModel
from .networks import Generator, Discriminator, CycleGANLoss
from .networks import get_norm_layer, ReplayBuffer, PathLengthPenalty

class CycleGAN(BaseModel):
    """
    CycleGAN model for image-to-image translation.

    Args:
    - input_nc: Number of input channels. Default is 3.
    - output_nc: Number of output channels. Default is 3.
    - n_residual_blocks: Number of residual blocks in generators. Default is 9.
    - n_features: Number of features in generators and discriminators. Default is 64.
    - n_downsampling: Number of downsampling layers in generators. Default is 2.
    - norm_type: Normalization layer type: 'batch', 'instance' or 'none'. Default is 'instance'.
    - add_skip: If True, add skip connections to the generators. Default is False.
    - use_replay_buffer: If True, use a replay buffer for adversarial loss. Default is False.
    - replay_buffer_size: Size of the replay buffer. Default is 50.
    - vanilla_loss: If True, use BCEWithLogitsLoss. Otherwise, use MSELoss. Default is True.
    - cycle_loss_weight: Weight for cycle-consistency loss. Default is 10.
    - id_loss_weight: Weight for identity loss. Default is 5.
    - plp_loss_weight: Weight for PLP loss. Default is 1.
    - plp_step: Steps between Path Length Penalty calculations. If zero, never use. Default is 0.
    - plp_beta: Beta for Path Length Penalty. Default is 0.99.
    - lr: Learning rate. Default is 0.0002.
    - beta1: Beta1 for Adam optimizer. Default is 0.5.
    - beta2: Beta2 for Adam optimizer. Default is 0.999.
    - amp: If True, use automatic mixed precision. Default is False.
    - device: 'cuda' or 'cpu'. Default is 'cpu'.
    """
    def __init__(self, input_nc=3, output_nc=3,
                 n_residual_blocks=9, n_features=64, n_downsampling=2,
                 norm_type='instance',
                 add_skip=False,
                 use_replay_buffer=False,
                 replay_buffer_size=50,
                 vanilla_loss=True,
                 cycle_loss_weight=10.0, id_loss_weight=5.0, plp_loss_weight=1.0,
                 plp_step=0,
                 plp_beta=0.99,
                 amp=False,
                 lr=0.0002, beta1=0.5, beta2=0.999, device='cpu'):
        super().__init__(device)

        torch.cuda.empty_cache()
        gc.collect()

        norm_layer = get_norm_layer(norm_type)
        # Initialize generators and discriminators
        gen_params = {
            'input_nc': input_nc,
            'output_nc': output_nc,
            'n_residual_blocks': n_residual_blocks,
            'n_features': n_features,
            'n_downsampling': n_downsampling,
            'add_skip': add_skip,
            'norm_layer': norm_layer,
        }
        self.gen_AtoB = Generator(**gen_params).to(self.device)
        self.gen_BtoA = Generator(**gen_params).to(self.device)
        self.dis_A = Discriminator(input_nc, n_features, norm_layer).to(self.device)
        self.dis_B = Discriminator(input_nc, n_features, norm_layer).to(self.device)

        # Define losses
        self.adversarial_loss = CycleGANLoss(vanilla_loss=vanilla_loss).to(self.device)
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
        self.plp_loss_weight = plp_loss_weight

        if use_replay_buffer:
            self.buffer_fake_A = ReplayBuffer(replay_buffer_size).push_and_pop
            self.buffer_fake_B = ReplayBuffer(replay_buffer_size).push_and_pop
        else:
            self.buffer_fake_A = lambda x: x
            self.buffer_fake_B = lambda x: x

        self.plp_A = PathLengthPenalty(beta=plp_beta, step=plp_step, device=device)
        self.plp_B = PathLengthPenalty(beta=plp_beta, step=plp_step, device=device)

        self.amp = amp
        if self.amp:
            self.scaler = torch.amp.GradScaler()


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
        def _get_adv_loss(fake_A, fake_B):
            loss_AtoB = self.adversarial_loss(self.dis_B(fake_B), target_is_real=True)
            loss_BtoA = self.adversarial_loss(self.dis_A(fake_A), target_is_real=True)
            return loss_AtoB, loss_BtoA

        def _get_cycle_loss(real_A, real_B, fake_A, fake_B):
            if self.cycle_loss_weight > 0:
                loss_A = self.cycle_loss(self.gen_BtoA(fake_B), real_A)
                loss_B = self.cycle_loss(self.gen_AtoB(fake_A), real_B)
            else:
                loss_A = torch.tensor(0.0, device=self.device)
                loss_B = torch.tensor(0.0, device=self.device)
            return loss_A, loss_B

        def _get_id_loss(real_A, real_B):
            if self.id_loss_weight > 0:
                loss_A = self.identity_loss(self.gen_BtoA(real_A), real_A)
                loss_B = self.identity_loss(self.gen_AtoB(real_B), real_B)
            else:
                loss_A = torch.tensor(0.0, device=self.device)
                loss_B = torch.tensor(0.0, device=self.device)
            return loss_A, loss_B

        def _get_plp_loss(real_A, real_B, fake_A, fake_B):
            if self.plp_A.is_plp_step() and self.plp_loss_weight > 0:
                loss_A = self.plp_A(real_A, fake_B)
                loss_B = self.plp_B(real_B, fake_A)
            else:
                loss_A = torch.tensor(0.0, device=self.device)
                loss_B = torch.tensor(0.0, device=self.device)
            return loss_A, loss_B


        if self.plp_loss_weight > 0:
            ok_A = self.plp_A.is_plp_step(step_count=True)
            ok_B = self.plp_B.is_plp_step(step_count=True)
            if ok_A and ok_B:
                real_A.requires_grad_()
                real_B.requires_grad_()

        fake_B, fake_A = self.forward(real_A, real_B)
        loss_adv_AtoB, loss_adv_BtoA = _get_adv_loss(fake_A, fake_B)
        loss_cycle_A, loss_cycle_B = _get_cycle_loss(real_A, real_B, fake_A, fake_B)
        loss_id_A, loss_id_B = _get_id_loss(real_A, real_B)
        loss_plp_A, loss_plp_B = _get_plp_loss(real_A, real_B, fake_A, fake_B)


        loss_G_ad = loss_adv_AtoB + loss_adv_BtoA
        loss_G_cycle = loss_cycle_A + loss_cycle_B
        loss_G_id = loss_id_A + loss_id_B
        loss_G_plp = loss_plp_A + loss_plp_B

        loss_G = loss_G_ad
        loss_G += self.cycle_loss_weight * loss_G_cycle
        loss_G += self.id_loss_weight * loss_G_id
        loss_G += self.plp_loss_weight * loss_G_plp


        loss_real_A = self.adversarial_loss(self.dis_A(real_A), target_is_real=True)
        fake_A_ = self.buffer_fake_A(fake_A.detach())
        loss_fake_A = self.adversarial_loss(self.dis_A(fake_A_), target_is_real=False)
        loss_D_A = (loss_real_A + loss_fake_A) * 0.5

        loss_real_B = self.adversarial_loss(self.dis_B(real_B), target_is_real=True)
        fake_B_ = self.buffer_fake_B(fake_B.detach())
        loss_fake_B = self.adversarial_loss(self.dis_B(fake_B_), target_is_real=False)
        loss_D_B = (loss_real_B + loss_fake_B) * 0.5

        return Loss(
            loss_G=loss_G,
            loss_D_A=loss_D_A,
            loss_D_B=loss_D_B,
            loss_G_ad=loss_G_ad.detach(),
            loss_G_cycle=loss_G_cycle.detach(),
            loss_G_id=loss_G_id.detach(),
            loss_G_plp=loss_G_plp.detach()
        )


    def optimize(self, real_A, real_B): # pylint: disable=arguments-differ
        """
        Perform one optimization step for the generators and discriminators.
        """
        self.train()

        if self.amp:
            with torch.autocast(device_type="cuda"):
                loss = self.compute_loss(real_A, real_B)

            self.optimizer_G.zero_grad()
            self.scaler.scale(loss.loss_G).backward()
            self.scaler.step(self.optimizer_G)

            self.optimizer_D_A.zero_grad()
            self.scaler.scale(loss.loss_D_A).backward()
            self.scaler.step(self.optimizer_D_A)

            self.optimizer_D_B.zero_grad()
            self.scaler.scale(loss.loss_D_B).backward()
            self.scaler.step(self.optimizer_D_B)

            self.scaler.update()
        else:
            loss = self.compute_loss(real_A, real_B)

            self.optimizer_G.zero_grad()
            loss.loss_G.backward()
            self.optimizer_G.step()

            self.optimizer_D_A.zero_grad()
            loss.loss_D_A.backward()
            self.optimizer_D_A.step()

            self.optimizer_D_B.zero_grad()
            loss.loss_D_B.backward()
            self.optimizer_D_B.step()

        return loss


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


    def generate_samples(self, real_A, real_B, n_images=4):
        """
        Generate samples with real, fake, reconstructed and identity images.
        """
        real_A = real_A[:n_images]
        real_B = real_B[:n_images]

        self.eval()
        with torch.no_grad():
            fake_B, fake_A = self.forward(real_A, real_B)
            recovered_B, recovered_A = self.forward(fake_A, fake_B)
            id_B, id_A = self.forward(real_B, real_A) # pylint: disable=arguments-out-of-order

        imgs_A = torch.vstack([real_A, fake_B, recovered_A, id_A])
        imgs_B = torch.vstack([real_B, fake_A, recovered_B, id_B])

        return imgs_A, imgs_B
