"""Test metrics module."""

import time
import unittest
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from metrics import FID, LPIPS  # pylint: disable=all
from utils.data_loader import get_img_dataloader  # pylint: disable=all
from utils.utils import get_gpu_memory_usage  # pylint: disable=all

class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.use_cuda = True
        self.batch_size = 2048
        self.print_memory = True

        folder = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'nexet'
        train_A_csv = folder / 'input_A_train_filtered.csv'
        test_A_csv = folder / 'input_A_test_filtered.csv'
        train_B_csv = folder / 'input_B_train_filtered.csv'
        test_B_csv = folder / 'input_B_test_filtered.csv'

        train_A = get_img_dataloader(csv_file=train_A_csv, batch_size=self.batch_size)
        test_A = get_img_dataloader(csv_file=test_A_csv, batch_size=self.batch_size)
        train_B = get_img_dataloader(csv_file=train_B_csv, batch_size=self.batch_size)
        test_B = get_img_dataloader(csv_file=test_B_csv, batch_size=self.batch_size)

        self.train_A_imgs = next(iter(train_A))
        self.test_A_imgs = next(iter(test_A))
        self.train_B_imgs = next(iter(train_B))
        self.test_B_imgs = next(iter(test_B))

        test_A_turbo = get_img_dataloader(
            csv_file=test_A_csv,
            img_dir=Path(test_A_csv).parent / Path(test_A_csv).stem.replace('_test_filtered', '_turbo').replace('in', 'out'),
            batch_size=self.batch_size)
        test_B_turbo = get_img_dataloader(
            csv_file=test_B_csv,
            img_dir=Path(test_B_csv).parent / Path(test_B_csv).stem.replace('_test_filtered', '_turbo').replace('in', 'out'),
            batch_size=self.batch_size)

        self.test_A_turbo_imgs = next(iter(test_A_turbo))
        self.test_B_turbo_imgs = next(iter(test_B_turbo))

        self.use_cuda = self.use_cuda and torch.cuda.is_available()

        if self.use_cuda:
            print("Using CUDA")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.empty_cache()

    def test_fid(self):
        print("===============")
        print("FID calculation")
        print("===============")
        if self.print_memory:
            print(get_gpu_memory_usage("Initital memory usage", short_msg=True))

        fid = FID(dims=2048, cuda=self.use_cuda, batch_size=128)
        if self.print_memory:
            print(get_gpu_memory_usage("After model load", short_msg=True))

        start_time = time.time()
        fid_equal = fid.get(self.train_A_imgs, self.train_A_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of the same images: {fid_equal:0.3g} ({elapsed_time/fid._last_num_imgs*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_same_A = fid.get(self.train_A_imgs, self.test_A_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of only A images: {fid_same_A:0.3g} ({elapsed_time/fid._last_num_imgs*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_same_B = fid.get(self.train_B_imgs, self.test_B_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of only B images: {fid_same_B:0.3g} ({elapsed_time/fid._last_num_imgs*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_different = fid.get(self.train_A_imgs, self.train_B_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of A x B images: {fid_different:0.3g} ({elapsed_time/fid._last_num_imgs*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_turbo_B = fid.get(self.test_A_imgs, self.test_B_turbo_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of CycleGAN-turbo A transformed images: {fid_turbo_B:0.3g} ({elapsed_time/fid._last_num_imgs*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_turbo_A = fid.get(self.test_B_imgs, self.test_A_turbo_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID score of CycleGAN-turbo B transformed images: {fid_turbo_A:0.3g} ({elapsed_time/fid._last_num_imgs*1000:.3f} s/1000 images)")

        if self.print_memory:
            print(get_gpu_memory_usage("After FID calculation", short_msg=True))
        torch.cuda.empty_cache()
        if self.print_memory:
            print(get_gpu_memory_usage("After emptying cache", short_msg=True))


        self.assertLess(fid_equal, 1E-3, "FID for same images should be zero.")
        self.assertLess(fid_same_A, fid_different, "FID for images of the same class A should be lower than for images of different classes.")
        self.assertLess(fid_same_B, fid_different, "FID for images of the same class B should be lower than for images of different classes.")


    def test_lpips(self):
        print("=================")
        print("LPIPS calculation")
        print("=================")
        if self.print_memory:
            print(get_gpu_memory_usage("Initital memory usage", short_msg=True))

        lpips = LPIPS(cuda=self.use_cuda, batch_size=128)
        if self.print_memory:
            print(get_gpu_memory_usage("After model load", short_msg=True))


        start_time = time.time()
        lpips_equal = lpips.get(self.train_A_imgs, self.train_A_imgs)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of the same images: {lpips_equal.mean():0.3g} ± {lpips_equal.std():0.3g} ({elapsed_time/lpips._last_num_pairs*1000:.3f} s/1000 image pairs)")

        start_time = time.time()
        lpips_equal_pairs = lpips.get(self.train_A_imgs, self.train_A_imgs, all_pairs=True)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of the same group of images by pairs: {lpips_equal_pairs.mean():0.3g} ± {lpips_equal_pairs.std():0.3g} ({elapsed_time/lpips._last_num_pairs*1000:.3f} s/1000 image pairs)")

        n = min([len(self.train_A_imgs), len(self.test_A_imgs)])
        start_time = time.time()
        lpips_same_A = lpips.get(self.train_A_imgs[:n], self.test_A_imgs[:n])
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of only A images: {lpips_same_A.mean():0.3g} ± {lpips_same_A.std():0.3g} ({elapsed_time/lpips._last_num_pairs*1000:.3f} s/1000 image pairs)")

        n = min([len(self.train_B_imgs), len(self.test_B_imgs)])
        start_time = time.time()
        lpips_same_B = lpips.get(self.train_B_imgs[:n], self.test_B_imgs[:n])
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of only B images: {lpips_same_B.mean():0.3g} ± {lpips_same_B.std():0.3g} ({elapsed_time/lpips._last_num_pairs*1000:.3f} s/1000 image pairs)")

        n = min([len(self.train_A_imgs), len(self.train_B_imgs)])
        start_time = time.time()
        lpips_different = lpips.get(self.train_A_imgs[:n], self.train_B_imgs[:n])
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of A x B images: {lpips_different.mean():0.3g} ± {lpips_different.std():0.3g} ({elapsed_time/lpips._last_num_pairs*1000:.3f} s/1000 image pairs)")

        n = min([len(self.test_A_imgs), len(self.test_B_turbo_imgs)])
        start_time = time.time()
        lpips_turbo_A = lpips.get(self.test_A_imgs[:n], self.test_B_turbo_imgs[:n])
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of CycleGAN-turbo A transformed images: {lpips_turbo_A.mean():0.3g} ± {lpips_turbo_A.std():0.3g} ({elapsed_time/lpips._last_num_pairs*1000:.3f} s/1000 image pairs)")

        n = min([len(self.test_B_imgs), len(self.test_A_turbo_imgs)])
        start_time = time.time()
        lpips_turbo_B = lpips.get(self.test_B_imgs[:n], self.test_A_turbo_imgs[:n])
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of CycleGAN-turbo B transformed images: {lpips_turbo_B.mean():0.3g} ± {lpips_turbo_B.std():0.3g} ({elapsed_time/lpips._last_num_pairs*1000:.3f} s/1000 image pairs)")

        if self.print_memory:
            print(get_gpu_memory_usage("After LPIPS calculation", short_msg=True))
        torch.cuda.empty_cache()
        if self.print_memory:
            print(get_gpu_memory_usage("After emptying cache", short_msg=True))

        self.assertLess(lpips_equal.mean(), 1E-3, "LPIPS loss for same images should be zero.")
        self.assertLess(lpips_same_A.mean(), lpips_different.mean(), "LPIPS loss for images of the same A class should be lower than for images of different classes.")
        self.assertLess(lpips_same_B.mean(), lpips_different.mean(), "LPIPS loss for images of the same B class should be lower than for images of different classes.")

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestMetrics('test_fid'))
    suite.addTest(TestMetrics('test_lpips'))

    runner = unittest.TextTestRunner()
    runner.run(suite)