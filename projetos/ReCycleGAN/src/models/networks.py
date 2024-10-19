"""Module with network constructors and loss functions."""
import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block.

    Args:
    - in_features: Number of features.
    """
    def __init__(self, in_features):
        super().__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward pass through the residual block."""
        return x + self.conv_block(x)

class Generator(nn.Module):
    """
    Generator network.

    Args:
    - input_nc: Number of input channels.
    - output_nc: Number of output channels.
    - n_residual_blocks: Number of residual blocks. Default is 9.
    - n_features: Number of features. Default is 64.
    - n_downsampling: Number of downsampling layers. Default is 2.
    - add_skip: If True, add skip connections. Default is False.
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, n_features=64, n_downsampling=2, add_skip=False):
        super().__init__()

        self.initial_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, n_features, 7),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(inplace=True),
        )

        self.encoder = nn.ModuleList()
        for i in range(n_downsampling):
            n_feat = n_features * 2 ** i
            self.encoder.append(nn.Sequential(
                nn.Conv2d(n_feat, 2 * n_feat, 3, stride=2, padding=1),
                nn.InstanceNorm2d(2 * n_feat),
            ))

        n_feat = n_features * 2 ** n_downsampling
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(n_feat) for _ in range(n_residual_blocks)]
        )

        self.decoder = nn.ModuleList()
        for i in range(n_downsampling):
            n_feat = n_features * 2 ** (n_downsampling - i)
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(n_feat, n_feat // 2, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(n_feat // 2),
                nn.ReLU(inplace=True),
            ))

        self.final_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_features, output_nc, 7),
            nn.Tanh(),
        )

        self.add_skip = add_skip

    def forward(self, x):
        """Forward pass through the generator."""
        x = self.initial_layers(x)

        skips = []
        for layer in self.encoder:
            x = layer(x)
            if self.add_skip:
                skips.append(x)

        x = self.residual_blocks(x)

        if self.add_skip:
            for i, layer in enumerate(self.decoder):
                x = layer(x + skips[-(i+1)])
        else:
            for layer in self.decoder:
                x = layer(x)

        return self.final_layers(x)

class Discriminator(nn.Module):
    """
    Discriminator network.

    Args:
    - input_nc: Number of input channels.
    - n_features: Number of features. Default is 64.
    """
    def __init__(self, input_nc, n_features=64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, n_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            self.discriminator_block(n_features, 2 * n_features),
            self.discriminator_block(2 * n_features, 4 * n_features),
            self.discriminator_block(4 * n_features, 8 * n_features),

            nn.Conv2d(8 * n_features, 1, 4, padding=1)
        )

    def discriminator_block(self, input_dim, output_dim):
        """
        Returns downsampling layers of each discriminator block

        Args:
        - input_dim: Number of input channels.
        - output_dim: Number of output channels.
        """
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        """Forward pass through the discriminator."""
        x =  self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.shape[0], -1)

class CycleGANLoss(nn.Module):
    """Define different GAN objectives.

    The CycleGANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    Args:
    - target_real_label: Target label for real images. Default is 1.0.
    - target_fake_label: Target label for fake images. Default is 0.0.
    - vanilla_loss: If True, use BCEWithLogitsLoss. Otherwise, use MSELoss. Default is True.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0, vanilla_loss=True):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if vanilla_loss:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.MSELoss()

    def _get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self._get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

    def forward(self, x):
        """Forward pass through the CycleGANLoss."""
