"""Test metrics module."""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from utils.data_loader import get_img_dataloader  # pylint: disable=all
from utils import show_img, image_folder_to_tensor, remove_all_files  # pylint: disable=all

class TestImageDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Load image paths from CSV files
        folder = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'nexet'
        self.train_A_csv = folder / 'input_A_train_filtered.csv'
        self.test_A_csv = folder / 'input_A_test_filtered.csv'
        self.train_B_csv = folder / 'input_B_train_filtered.csv'
        self.test_B_csv = folder / 'input_B_test_filtered.csv'

        self.out_folder = Path(__file__).resolve().parent.parent / 'no_sync/imgs'
        self.out_folder.mkdir(parents=True, exist_ok=True)
        remove_all_files(self.out_folder)

    def test_dataloader(self):
        """Test loading images."""
        train_A = get_img_dataloader(
            csv_file=self.train_A_csv,
            img_dir=Path(self.train_A_csv).parent / Path(self.train_A_csv).stem.replace('_train_filtered', ''),
            transformation=None,
            file_name_col='file_name',
            batch_size=32,
            shuffle=True,
            num_workers=4)

        n_imgs = len(pd.read_csv(self.train_A_csv)) - 1
        n_batches = np.ceil(n_imgs / 32)
        self.assertEqual(len(train_A), n_batches, f'There should be {n_batches} batches.')

        imgs_A = next(iter(train_A))

        self.assertGreaterEqual(torch.min(imgs_A), 0.0, 'Image values should be greater than or equal to 0.')
        self.assertLessEqual(torch.max(imgs_A), 1.0, 'Image values should be less than or equal to 1.')

        shp = imgs_A.shape
        self.assertEqual(shp[0], 32, 'Batch size should be 32.')
        self.assertEqual(shp[1], 3, 'Number of channels should be 3.')
        self.assertEqual(shp[2], 256, 'Image height should be 256.')
        self.assertEqual(shp[3], 256, 'Image width should be 256.')

    def test_save_images(self):
        """Test saving images."""
        train_A = get_img_dataloader(self.train_A_csv)

        imgs_A = next(iter(train_A))
        show_img(imgs_A, title='Train A images', figsize = (10, 6))
        test_file = self.out_folder / 'test_dataloader_A.png'
        plt.savefig(test_file)
        self.assertTrue(test_file.exists(), f"File {test_file.name} does not exist")

        train_B = get_img_dataloader(self.train_B_csv)

        imgs_B = next(iter(train_B))
        show_img(imgs_B, title='Train B images', figsize = (10, 6))
        test_file = self.out_folder / 'test_dataloader_B.png'
        plt.savefig(test_file)
        self.assertTrue(test_file.exists(), f"File {test_file.name} does not exist")

        show_img(torch.vstack([imgs_A[:4], imgs_B[:4]]), title='', figsize = (10, 6))
        test_file = self.out_folder / 'nexet_imgs.png'
        plt.savefig(test_file)
        self.assertTrue(test_file.exists(), f"File {test_file.name} does not exist")


        turbo_A = get_img_dataloader(
            csv_file=self.test_B_csv,
            img_dir=Path(self.test_B_csv).parent / Path(self.test_B_csv).stem.replace('_test_filtered', '_turbo').replace('in', 'out'))
        imgs_A = next(iter(turbo_A))
        turbo_B = get_img_dataloader(
            csv_file=self.test_A_csv,
            img_dir=Path(self.test_A_csv).parent / Path(self.test_A_csv).stem.replace('_test_filtered', '_turbo').replace('in', 'out'))
        imgs_B = next(iter(turbo_B))

        show_img(torch.vstack([imgs_A[:4], imgs_B[:4]]), title='', figsize = (10, 6))
        test_file = self.out_folder / 'turbo_imgs.png'
        plt.savefig(test_file)
        self.assertTrue(test_file.exists(), f"File {test_file.name} does not exist")

    def test_dataloader_new_range(self):
        """Test dataloader with [-1,1] range of values."""
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_A = get_img_dataloader(
            csv_file=self.train_A_csv,
            transformation=transformation)

        imgs_A = next(iter(train_A))

        self.assertGreaterEqual(torch.min(imgs_A), -1.0, 'Image values should be greater than or equal to -1.')
        self.assertLess(torch.min(imgs_A), 0.0, 'Image values can be less than 0.')
        self.assertLessEqual(torch.max(imgs_A), 1.0, 'Image values should be less than or equal to 1.')

        show_img(imgs_A, title='Train A images ±1 scale', figsize = (10, 6))
        plt.savefig(self.out_folder / 'test_dataloader_A_new_scale.png')

    def test_image_folder_to_tensor(self):
        """Test image_folder_to_tensor function."""
        img_dir = Path(__file__).resolve().parent.parent / 'docs/assets/haze'
        imgs_A = image_folder_to_tensor(img_dir, img_glob='*_A.jpg')
        imgs_B = image_folder_to_tensor(img_dir, img_glob='*_B.jpg')

        shp = imgs_A.shape
        self.assertEqual(shp[0], 4, 'Batch size should be 4.')
        shp = imgs_B.shape
        self.assertEqual(shp[0], 4, 'Batch size should be 4.')

        show_img(torch.vstack([imgs_A, imgs_B]), title='', figsize = (10, 6))
        plt.savefig(self.out_folder / 'haze_imgs.png')


if __name__ == '__main__':
    unittest.main()
