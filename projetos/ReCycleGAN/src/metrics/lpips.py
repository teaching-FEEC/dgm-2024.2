"""LPIPS wrapper.

Source: https://github.com/richzhang/PerceptualSimilarity/"""

import warnings
import torch
import lpips
import numpy as np

class LPIPS():
    """Wrapper to compute Perceptual Similarity Metric (LPIPS).

    Attributes
    ----------
    net: str
        Network to use. Can be 'alex', 'squeeze' or 'vgg'.
        (Default: 'alex')
    cuda : bool
        If True, use GPU to compute activations.
        (Default: False).
    rescale : bool
        If True, rescale images from [0,1] to [-1,1].
        LPIPS expects images in [-1,1].
        (Default: True).
    no_grad : bool
        If True, use torch.no_grad() context.
        (Default: True).
    """
    def __init__(self, net='alex', cuda=False, rescale=True, no_grad=True):
        self.cuda = cuda
        self.rescale = rescale
        self.no_grad = no_grad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lpips.LPIPS(net=net, version='0.1')
        if no_grad:
            self.model.eval()
        if cuda:
            self.model.cuda()

    @staticmethod
    def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
        """Convert tensor to image."""
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
        return image_numpy.astype(imtype)

    @staticmethod
    def im2tensor(image, cent=1., factor=255./2.):
        """Convert image to tensor."""
        return torch.Tensor((image / factor - cent)
                            [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

    def get(self, images_real,images_fake):
        """Calculate LPIPS between real and fake images."""
        if self.cuda:
            images_real = images_real.cuda()
            images_fake = images_fake.cuda()

        if self.no_grad:
            with torch.no_grad():
                lpips_value = self.model.forward(images_real, images_fake, normalize=self.rescale)
        else:
            lpips_value = self.model.forward(images_real, images_fake)
        return lpips_value
