# pylint: disable=line-too-long
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
        (Default: 32).
    max_pairs : int
        Maximum number of pairs to use.
        (Default: 10000).
    """
    def __init__(self, net='alex', cuda=False, rescale=True, no_grad=True, batch_size=32, max_pairs=10000):
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

    def get_last_num_pairs(self):
        """Return the number of pairs used in the last call to get."""
        return self._last_num_pairs

    def _lpips_dataloader(self, img1, img2, normalize, use_all_pairs):
        if use_all_pairs:
            n_max = self.max_pairs
        else:
            n_max = len(img1.dataset)

        n = 0
        pred_arr = torch.empty(0)
        while True:
            for batch1,batch2 in tqdm(zip(img1,img2)):

                if self.cuda:
                    batch1 = batch1.cuda()
                    batch2 = batch2.cuda()

                if self.no_grad:
                    with torch.no_grad():
                        lpips_values = self.model.forward(batch1, batch2, normalize=normalize)
                else:
                    lpips_values = self.model.forward(batch1, batch2, normalize=normalize)

                pred_arr = torch.cat((pred_arr, lpips_values.cpu()))
                n += len(batch1)
                if n >= n_max:
                    self._last_num_pairs = n
                    return pred_arr


    def _lpips(self, img0, img1, normalize, use_all_pairs):
        if use_all_pairs:
            all_pairs = self._get_all_pairs(img0, img1)
        else:
            all_pairs = np.array([(i, i) for i in range(len(img0))])
        self._last_num_pairs = len(all_pairs)

        pred_arr = torch.empty(0)
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

            pred_arr = torch.cat((pred_arr, lpips_values))
            start_idx += self.batch_size
        return pred_arr

    def _get_all_pairs(self, list1, list2):
        i1 = list(range(len(list1)))
        i2 = list(range(len(list2)))
        all_pairs = list(itertools.product(i1, i2))
        random.shuffle(all_pairs)
        return np.array(all_pairs[:min(len(all_pairs), self.max_pairs)])

    def get(self, images1,images2, all_pairs=False):
        """Calculate LPIPS between pairs of images."""

        if isinstance(images2, torch.utils.data.DataLoader) != isinstance(images1, torch.utils.data.DataLoader):
            msg = 'Both images must be either DataLoader or list.'
            raise ValueError(msg)

        if not all_pairs:
            if isinstance(images1, torch.utils.data.DataLoader):
                n_imgs1 = np.sum([len(batch) for batch in images1])
                batch_size1 = images1.batch_size
                n_imgs2 = np.sum([len(batch) for batch in images2])
                batch_size2 = images2.batch_size
            else:
                n_imgs1 = len(images1)
                batch_size1 = self.batch_size
                n_imgs2 = len(images2)
                batch_size2 = self.batch_size

            if n_imgs1 != n_imgs2:
                msg = 'Number of images per group must be the same.'
                raise ValueError(msg)
            if batch_size1 != batch_size2:
                msg = 'Batch sizes must be the same.'
                raise ValueError(msg)

        if isinstance(images1, torch.utils.data.DataLoader):
            return self._lpips_dataloader(images1, images2, normalize=self.rescale, use_all_pairs=all_pairs)
        return self._lpips(images1, images2, normalize=self.rescale, use_all_pairs=all_pairs)
