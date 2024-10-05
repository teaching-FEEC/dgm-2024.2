"""Test metrics module."""

import gc
import time
import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from metrics import FID, LPIPS  # pylint: disable=all
from utils.data_loader import get_img_dataloader  # pylint: disable=all
from utils.utils import print_gpu_memory_usage  # pylint: disable=all

class TestFID(unittest.TestCase):
    def setUp(self):
        # Use cuda
        self.cuda = True
        # Load image paths from CSV files
        self.n = 200
        # Print memory usage
        self.print_memory = True

        folder = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'nexet'
        self.train_A_csv = folder / 'input_A_train.csv'
        self.test_A_csv = folder / 'input_A_test.csv'
        self.train_B_csv = folder / 'input_B_train.csv'
        self.test_B_csv = folder / 'input_B_test.csv'

        train_A = get_img_dataloader(
            csv_file=self.train_A_csv,
            img_dir=Path(self.train_A_csv).parent / Path(self.train_A_csv).stem.replace('_train', ''),
            batch_size=self.n)
        test_A = get_img_dataloader(
            csv_file=self.test_A_csv,
            img_dir=Path(self.test_A_csv).parent / Path(self.test_A_csv).stem.replace('_test', ''),
            batch_size=self.n)
        train_B = get_img_dataloader(
            csv_file=self.train_B_csv,
            img_dir=Path(self.train_B_csv).parent / Path(self.train_B_csv).stem.replace('_train', ''),
            batch_size=self.n)
        test_B = get_img_dataloader(
            csv_file=self.test_B_csv,
            img_dir=Path(self.test_B_csv).parent / Path(self.test_B_csv).stem.replace('_test', ''),
            batch_size=self.n)

        self.train_A_imgs = next(iter(train_A))
        self.test_A_imgs = next(iter(test_A))
        self.train_B_imgs = next(iter(train_B))
        self.test_B_imgs = next(iter(test_B))

        if self.cuda:
            self.train_A_imgs = self.train_A_imgs.cuda()
            self.test_A_imgs = self.test_A_imgs.cuda()
            self.train_B_imgs = self.train_B_imgs.cuda()
            self.test_B_imgs = self.test_B_imgs.cuda()


    def test_fid(self):
        print("===============")
        print("FID calculation")
        print("===============")
        if self.print_memory:
            print_gpu_memory_usage("Initital memory usage:")
        fid = FID(dims=2048, cuda=self.cuda)

        start_time = time.time()
        fid_equal = fid.get(self.train_A_imgs, self.train_A_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of the same images: {fid_equal:0.3g} ({elapsed_time:.3f} s)")

        start_time = time.time()
        fid_same_A = fid.get(self.train_A_imgs, self.test_A_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of only A images: {fid_same_A:0.3g} ({elapsed_time:.3f} s)")

        start_time = time.time()
        fid_same_B = fid.get(self.train_B_imgs, self.test_B_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of only B images: {fid_same_B:0.3g} ({elapsed_time:.3f} s)")

        start_time = time.time()
        fid_different = fid.get(self.train_A_imgs, self.train_B_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of A x B images: {fid_different:0.3g} ({elapsed_time:.3f} s)")

        if self.print_memory:
            print_gpu_memory_usage("After FID calculation:")

        fid = None
        torch.cuda.empty_cache()
        gc.collect()

        if self.print_memory:
            print_gpu_memory_usage("After garbage collection:")

        self.assertLess(fid_equal, 1E-3, "FID for same images should be zero.")
        self.assertLess(fid_same_A, fid_different, "FID for images of the same class A should be lower than for images of different classes.")
        self.assertLess(fid_same_B, fid_different, "FID for images of the same class B should be lower than for images of different classes.")


    def test_lpips(self):
        print("=================")
        print("LPIPS calculation")
        print("=================")

        if self.print_memory:
            print_gpu_memory_usage("Initital memory usage:")
        lpips = LPIPS(cuda=self.cuda)

        start_time = time.time()
        lpips_equal = lpips.get(self.train_A_imgs, self.train_A_imgs)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of the same images: {lpips_equal.mean():0.3g} ± {lpips_equal.std():0.3g} ({elapsed_time:.3f} s)")

        start_time = time.time()
        lpips_same_A = lpips.get(self.train_A_imgs, self.test_A_imgs)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of only A images: {lpips_same_A.mean():0.3g} ± {lpips_same_A.std():0.3g} ({elapsed_time:.3f} s)")

        start_time = time.time()
        lpips_same_B = lpips.get(self.train_A_imgs, self.test_A_imgs)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of only B images: {lpips_same_B.mean():0.3g} ± {lpips_same_B.std():0.3g} ({elapsed_time:.3f} s)")

        start_time = time.time()
        lpips_different = lpips.get(self.train_A_imgs, self.train_B_imgs)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of A x B images: {lpips_different.mean():0.3g} ± {lpips_different.std():0.3g} ({elapsed_time:.3f} s)")

        if self.print_memory:
            print_gpu_memory_usage("After LPIPS calculation:")

        lpips = None
        torch.cuda.empty_cache()
        gc.collect()

        if self.print_memory:
            print_gpu_memory_usage("After garbage collection:")

        self.assertLess(lpips_equal.mean(), 1E-3, "LPIPS loss for same images should be zero.")
        self.assertLess(lpips_same_A.mean(), lpips_different.mean(), "LPIPS loss for images of the same A class should be lower than for images of different classes.")
        self.assertLess(lpips_same_B.mean(), lpips_different.mean(), "LPIPS loss for images of the same B class should be lower than for images of different classes.")

if __name__ == '__main__':
    unittest.main()