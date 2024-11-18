import argparse
import glob
import os
import shutil

import numpy as np
from cityscapesscripts.download.downloader import download_packages, login
from PIL import Image
from scipy.io import loadmat
from torch import nn
from torch.utils import data
from torchvision.transforms import v2
from tqdm.auto import tqdm


class SemanticResize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.x_transform = v2.Resize(size=size)
        self.s_transform = v2.Resize(size=size, interpolation=v2.InterpolationMode.NEAREST_EXACT)

    def forward(self, sample):
        return {'x': self.x_transform(sample['x']), 's': self.s_transform(sample['s'])}


class SemanticData(data.Dataset):
    def __init__(self, folder_path, crop_shape, transform=None, x_transform=None, s_transform=None):
        super().__init__()
        self.transform = transform
        self.x_transform = x_transform
        self.s_transform = s_transform
        self.img_files = glob.glob(os.path.join(folder_path, 'images', '*.png'))
        self.img_files += glob.glob(os.path.join(folder_path, 'images', '*.jpg'))
        fns = [os.path.splitext(os.path.basename(img_path))[0] for img_path in self.img_files]
        self.map_files = [os.path.join(folder_path, 'maps', fn + '.png') for fn in fns]
        self.data = [v2.CenterCrop(size=crop_shape)(Image.open(img_path).convert('RGB')) for img_path in self.img_files]
        self.labels = [v2.CenterCrop(size=crop_shape)(Image.open(map_path)) for map_path in self.map_files]

    def __getitem__(self, index):
            sample = {'x': self.data[index], 's': self.labels[index]}
            if self.transform:
                sample = self.transform(sample)
            if self.x_transform:
                sample['x'] = self.x_transform(sample['x'])
            if self.s_transform:
                sample['s'] = self.s_transform(sample['s'])
            return sample

    def __len__(self):
        return len(self.img_files)


def download_cityscapes():
    session = login()
    download_packages(session=session, package_names=["leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip"], destination_path="data/", resume=True)

def organize_split(root, fnames):
    map_path = "data/coco/annotations"
    img_path = "data/coco/images"
    for i,fname in tqdm(enumerate(fnames)):
        s = loadmat(os.path.join(map_path, fname) + '.mat')['S']
        fn = os.path.join(root, "semantic", fname)
        Image.fromarray(s).save(f"{fn}.png")
        shutil.move(os.path.join(img_path, fname) + '.jpg', os.path.join(root, "images"))

def process_cocostuff():
    in_path = "data/coco/annotations"
    train_path = "data/coco/train"
    val_path = "data/coco/val"
    with open("data/coco/imageLists/train.txt") as f:
        fnames = f.read().splitlines()
    organize_split(train_path, fnames)
    with open("data/coco/imageLists/test.txt") as f:
        fnames = f.read().splitlines()
    organize_split(val_path, fnames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset (city/coco)')
    args = parser.parse_args()
    if args.dataset.lower() == "city":
        download_cityscapes()
    elif args.dataset.lower() == "coco":
        process_cocostuff()
    