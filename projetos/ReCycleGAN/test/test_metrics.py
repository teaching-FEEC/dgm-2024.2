"""Test metrics module."""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from metrics import FID  # pylint: disable=all
from utils.data_loader import get_img_dataloader  # pylint: disable=all

class TestFID(unittest.TestCase):
    def setUp(self):
        # Load image paths from CSV files
        folder = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'nexet'
        self.train_A_csv = folder / 'input_A_train.csv'
        self.test_A_csv = folder / 'input_A_test.csv'
        self.train_B_csv = folder / 'input_B_train.csv'
        self.test_B_csv = folder / 'input_B_test.csv'

        train_A = get_img_dataloader(
            csv_file=self.train_A_csv,
            img_dir=Path(self.train_A_csv).parent / Path(self.train_A_csv).stem.replace('_train', ''),
            batch_size=100)
        train_B = get_img_dataloader(
            csv_file=self.train_B_csv,
            img_dir=Path(self.train_B_csv).parent / Path(self.train_B_csv).stem.replace('_train', ''),
            batch_size=100)

        self.train_A_imgs = next(iter(train_A))
        self.train_B_imgs = next(iter(train_B))

    def test_fid(self):
        # Run FID calculation
        n = 20

        fid = FID(dims=2048, cuda=True)

        fid_equal = fid.get(self.train_A_imgs[:n], self.train_A_imgs[:n])
        print(f"FID score of the same images: {fid_equal}")

        fid_same = fid.get(self.train_A_imgs[:n], self.train_A_imgs[-n:])
        print(f"FID score of only A images: {fid_same}")

        fid_different = fid.get(self.train_A_imgs[:n], self.train_B_imgs[:n])
        print(f"FID score of A x B images: {fid_different}")

        self.assertGreater(1E-3, fid_equal, "FID for same images should be zero.")
        self.assertGreater(fid_different, fid_same, "FID for images of the same class should be lower than for images of different classes.")

if __name__ == '__main__':
    unittest.main()