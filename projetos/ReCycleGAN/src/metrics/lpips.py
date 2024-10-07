"""LPIPS wrapper.

Source: https://github.com/richzhang/PerceptualSimilarity/"""

import random
import math
import itertools
import warnings
import torch
import lpips
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        """Dummy function for tqdm."""
        return x

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
    batch_size : int
        Batch size to use.
        (Default: 128).
    max_pairs : int
        Maximum number of pairs to use.
        (Default: 10000).
    """
    def __init__(self, net='alex', cuda=False, rescale=True, no_grad=True, batch_size=128, max_pairs=10000):
        self.cuda = cuda
        self.rescale = rescale
        self.no_grad = no_grad
        self.batch_size = batch_size
        self.max_pairs = max_pairs
        self._last_num_pairs = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lpips.LPIPS(net=net, version='0.1')
        if no_grad:
            self.model.eval()
        if cuda:
            self.model.cuda()

    def _lpips(self, img0, img1, normalize, use_all_pairs):
        if use_all_pairs:
            all_pairs = self._get_all_pairs(img0, img1)
        else:
            all_pairs = np.array([(i, i) for i in range(len(img0))])
        self._last_num_pairs = len(all_pairs)

        pred_arr = None
        start_idx = 0
        for _ in tqdm(range(math.ceil(len(all_pairs) / self.batch_size))):
            i0 = all_pairs[start_idx:start_idx + self.batch_size, 0]
            i1 = all_pairs[start_idx:start_idx + self.batch_size, 1]

            if self.cuda:
                img0_ = img0[i0].cuda()
                img1_ = img1[i1].cuda()
            else:
                img0_ = img0[i0]
                img1_ = img1[i1]

            if self.no_grad:
                with torch.no_grad():
                    lpips_values = self.model.forward(img0_, img1_, normalize=normalize)
            else:
                lpips_values = self.model.forward(img0_, img1_, normalize=normalize)

            if pred_arr is None:
                pred_arr = lpips_values
            else:
                pred_arr = torch.cat((pred_arr, lpips_values))
            start_idx += self.batch_size
        return pred_arr

    def _get_all_pairs(self, list1, list2):
        i1 = list(range(len(list1)))
        i2 = list(range(len(list2)))
        all_pairs = list(itertools.product(i1, i2))
        random.shuffle(all_pairs)
        return np.array(all_pairs[:min(len(all_pairs), self.max_pairs)])

    def get(self, images_real,images_fake, all_pairs=False):
        """Calculate LPIPS between real and fake images."""
        if not all_pairs:
            if len(images_real) != len(images_fake):
                msg = 'Number of real and fake images must be the same.'
                raise ValueError(msg)

        return self._lpips(images_real, images_fake, normalize=self.rescale, use_all_pairs=all_pairs)
