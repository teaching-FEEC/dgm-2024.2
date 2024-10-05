"""FID wrapper.

Source: https://github.com/mseitzer/pytorch-fid"""

import math
import torch
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        """Dummy function for tqdm."""
        return x

class FID():
    """Wrapper to compute Fréchet Inception Distance (FID).

    Attributes
    ----------
    dims : int
        Dimensionality of features in InceptionV3 network.
        (Default: 2048)
    cuda : bool
        If True, use GPU to compute activations.
        (Default: False).
    init_model : bool
        If True, initialize InceptionV3 model.
        (Default: True).
    batch_size : int
        Batch size to use.
        (Default: 128).
    """
    def __init__(self, dims=2048, cuda=False, init_model=True, batch_size=128):
        self.cuda = cuda
        self.dims = dims
        self.batch_size = batch_size

        self.model = None
        if init_model:
            self._init_model()

    def _init_model(self):
        if self.model is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            self.model = InceptionV3([block_idx])
            if self.cuda:
                self.model = self.model.to('cuda')
            self.model.eval()

    def _get_activations(self, imgs):
        pred_arr = np.empty((len(imgs), self.dims))

        start_idx = 0

        for _ in tqdm(range(math.ceil(len(imgs) / self.batch_size))):
            batch = imgs[start_idx:start_idx + self.batch_size]
            with torch.no_grad():
                pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        return pred_arr

    def _compute_statistics_of_imgs(self, imgs):
        act = self._get_activations(imgs)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def get(self, images_real, images_fake):
        """Calculate FID between real and fake images."""
        if self.cuda:
            images_real = images_real.cuda()
            images_fake = images_fake.cuda()
        self._init_model()

        m1, s1 = self._compute_statistics_of_imgs(images_real)
        m2, s2 = self._compute_statistics_of_imgs(images_fake)
        return fid_score.calculate_frechet_distance(m1, s1, m2, s2)

    def get_from_paths(self, path_images_real, path_images_fake):
        """Calculate FID between real and fake images."""
        if self.cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return fid_score.calculate_fid_given_paths(
            [path_images_real, path_images_fake],
            batch_size=self.batch_size,
            device=device,
            dims=self.dims,
            num_workers=4)
