"""Test metrics module."""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
# from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from utils.data_loader import get_img_dataloader  # pylint: disable=all
from utils import show_img  # pylint: disable=all

class TestImageDataLoader(unittest.TestCase):

    def setUp(self):
        # Load image paths from CSV files
        folder = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'nexet'
        self.train_A_csv = folder / 'input_A_train.csv'
        self.test_A_csv = folder / 'input_A_test.csv'
        self.train_B_csv = folder / 'input_B_train.csv'
        self.test_B_csv = folder / 'input_B_test.csv'

    def test_dataloader(self):

        train_A = get_img_dataloader(
            csv_file=self.train_A_csv,
            img_dir=Path(self.train_A_csv).parent / Path(self.train_A_csv).stem.replace('_train', ''),
            transformation=None,
            file_name_col='file_name',
            batch_size=32,
            shuffle=True,
            num_workers=4)

        n_imgs = len(pd.read_csv(self.train_A_csv)) - 1
        n_batches = np.ceil(n_imgs / 32)
        self.assertEqual(len(train_A), n_batches, f'There should be {n_batches} batches.')

        imgs = next(iter(train_A))

        self.assertGreaterEqual(torch.min(imgs), 0.0, 'Image values should be greater than or equal to 0.')
        self.assertGreaterEqual(1.0, torch.max(imgs), 'Image values should be lower than or equal to 1.')

        shp = imgs.shape
        self.assertEqual(shp[0], 32, 'Batch size should be 32.')
        self.assertEqual(shp[1], 3, 'Number of channels should be 3.')
        self.assertEqual(shp[2], 256, 'Image height should be 256.')
        self.assertEqual(shp[3], 256, 'Image width should be 256.')

        show_img(imgs, title='Train A images', figsize = (10, 6), show=False)
        plt.savefig('test_dataloader_A.png')

        train_B = get_img_dataloader(
            csv_file=self.train_B_csv,
            img_dir=Path(self.train_B_csv).parent / Path(self.train_B_csv).stem.replace('_train', ''))

        imgs = next(iter(train_B))
        show_img(imgs, title='Train B images', figsize = (10, 6), show=False)
        plt.savefig('test_dataloader_B.png')



if __name__ == '__main__':
    unittest.main()