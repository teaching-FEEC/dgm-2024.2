import torch
from torch import nn


def encoder_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1, dilation=1, bias=True),
                         nn.BatchNorm2d(num_features=out_dim),
                         nn.LeakyReLU(negative_slope=0.2, inplace=True))


def decoder_block(in_dim, out_dim):
    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1),
                         nn.BatchNorm2d(num_features=out_dim),
                         nn.Dropout(p=0.5),
                         nn.ReLU(inplace=True))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = encoder_block(in_dim=64, out_dim=128)
        self.conv3 = encoder_block(in_dim=128, out_dim=256)
        self.conv4 = encoder_block(in_dim=256, out_dim=512)
        self.conv5 = encoder_block(in_dim=512, out_dim=512)
        self.conv6 = encoder_block(in_dim=512, out_dim=512)
        self.conv7 = encoder_block(in_dim=512, out_dim=512)
        self.conv8 = encoder_block(in_dim=512, out_dim=512)
        self.deconv8 = decoder_block(in_dim=512, out_dim=512)
        self.deconv7 = decoder_block(in_dim=512, out_dim=512)
        self.deconv6 = decoder_block(in_dim=512, out_dim=512)
        self.deconv5 = decoder_block(in_dim=512, out_dim=512)
        self.deconv4 = decoder_block(in_dim=512, out_dim=256)
        self.deconv3 = decoder_block(in_dim=256, out_dim=128)
        self.deconv2 = decoder_block(in_dim=128, out_dim=64)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1),
                                     nn.Tanh())

    def forward(self, x):
        enc1 = self.conv1(x)  # 64
        enc2 = self.conv2(enc1)  # 128
        enc3 = self.conv3(enc2)  # 256
        enc4 = self.conv4(enc3)  # 512
        enc5 = self.conv5(enc4)  # 512
        enc6 = self.conv6(enc5)  # 512
        enc7 = self.conv7(enc6)  # 512
        enc8 = self.conv8(enc7)  # 512
        dec8 = self.deconv8(enc8)+enc7
        dec7 = self.deconv7(dec8)+enc6
        dec6 = self.deconv6(dec7)+enc5
        dec5 = self.deconv5(dec6)+enc4
        dec4 = self.deconv4(dec5)+enc3
        dec3 = self.deconv3(dec4)+enc2
        dec2 = self.deconv2(dec3)+enc1
        dec1 = self.deconv1(dec2)
        return dec1

    def get_gen(self):
        return self.gen


def block_discriminator(in_dim, out_dim,stride_size,use_batchnorm):
    if use_batchnorm is True:
        return nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=4, stride=stride_size, padding=1, dilation=1, bias=True),
                            nn.BatchNorm2d(num_features=out_dim),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=4, stride=stride_size, padding=1, dilation=1, bias=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = block_discriminator(in_dim=2, out_dim=64,stride_size=2,use_batchnorm=False)
        self.conv2 = block_discriminator(in_dim=64, out_dim=128,stride_size=2,use_batchnorm=True)
        self.conv3 = block_discriminator(in_dim=128, out_dim=256,stride_size=2,use_batchnorm=True)
        self.conv4 = block_discriminator(in_dim=256, out_dim=512,stride_size=1,use_batchnorm=True)
        #self.conv5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, dilation=1, bias=True),
        #                           nn.Sigmoid())
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, dilation=1, bias=True))

    def forward(self, x, y):
        xy_concat = torch.cat([y, x], dim=1)
        xy_concat = self.conv1(xy_concat)
        xy_concat = self.conv2(xy_concat)
        xy_concat = self.conv3(xy_concat)
        xy_concat = self.conv4(xy_concat)
        out = self.conv5(xy_concat)
        return out

    def get_disc(self):
        return self.disc
