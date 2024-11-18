import torch
from utils import add_uniform_noise, add_gaussian_noise

class AddUniformNoise(object):
    def __init__(self, lung_area=False,intensity=1):
        self.lung_area = lung_area
        self.intensity = intensity
    
    def __call__(self, tensor):
        return add_uniform_noise(tensor=tensor, intensity=self.intensity, lung_area=self.lung_area)
    
    def __repr__(self):
        if self.lung_area is True:
            return self.__class__.__name__ + 'uniform noise only at masked region'
        else:
            return self.__class__.__name__ + 'uniform noise at the whole image'


class AddGaussianNoise(object):
    # Modified from ptrblck suggestion at:
    # https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
    def __init__(self, mean=0., std=1.,lung_area=False,intensity=1):
        self.std = std
        self.mean = mean
        self.lung_area = lung_area
        self.intensity = intensity
        
    def __call__(self, tensor):
        return add_gaussian_noise(tensor, mean=self.mean, std=self.std, intensity=self.intensity, lung_area=self.lung_area)
    
    def __repr__(self):
        if self.lung_area is True:
            return self.__class__.__name__ + 'Gaussian noise with (mean={0}, std={1})'.format(self.mean, self.std) + ' only at masked region'
        else:
            return self.__class__.__name__ + 'Gaussian noise with (mean={0}, std={1})'.format(self.mean, self.std) + ' at the whole image'  