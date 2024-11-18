# pylint: disable=import-error, wrong-import-position,invalid-name,line-too-long
"""Test metrics module."""

import time
import unittest
import sys
from pathlib import Path
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from metrics.fid import FID
from metrics.lpips import LPIPS
from utils.data_loader import get_img_dataloader, copy_dataloader
from utils.utils import get_gpu_memory_usage

class TestMetrics(unittest.TestCase):
    """Test metrics module."""
    @classmethod
    def setUpClass(cls):
        cls.use_cuda = True
        cls.batch_size = 16
        cls.print_memory = True

        folder = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'nexet'
        train_A_csv = folder / 'input_A_train_filtered.csv'
        test_A_csv = folder / 'input_A_test_filtered.csv'
        train_B_csv = folder / 'input_B_train_filtered.csv'
        test_B_csv = folder / 'input_B_test_filtered.csv'

        cls.train_A = get_img_dataloader(csv_file=train_A_csv, batch_size=cls.batch_size)
        cls.test_A  = get_img_dataloader(csv_file=test_A_csv, batch_size=cls.batch_size)
        cls.train_B = get_img_dataloader(csv_file=train_B_csv, batch_size=cls.batch_size)
        cls.test_B  = get_img_dataloader(csv_file=test_B_csv, batch_size=cls.batch_size)

        cls.train_A_turbo = get_img_dataloader(
            csv_file=train_A_csv,
            img_dir=Path(train_A_csv).parent / Path(train_A_csv).stem.replace('_train_filtered', '_turbo').replace('in', 'out'),
            batch_size=cls.batch_size)
        cls.train_B_turbo = get_img_dataloader(
            csv_file=train_B_csv,
            img_dir=Path(train_B_csv).parent / Path(train_B_csv).stem.replace('_train_filtered', '_turbo').replace('in', 'out'),
            batch_size=cls.batch_size)

        cls.use_cuda = cls.use_cuda and torch.cuda.is_available()

        if cls.use_cuda:
            print("Using CUDA")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.empty_cache()

    def test_fid_dataloader(self):
        """Test FID calculation using dataloaders and images."""
        print("=======================")
        print("FID calculation options")
        print("=======================")

        if self.print_memory:
            print(get_gpu_memory_usage("Initital memory usage", short_msg=True))

        fid = FID(dims=2048, cuda=self.use_cuda, batch_size=16)
        if self.print_memory:
            print(get_gpu_memory_usage("After model load", short_msg=True))

        print("FID with data loaders")
        start_time = time.time()
        fid_dataloaders = fid.get(self.train_A, self.train_B_turbo)
        elapsed_time = time.time() - start_time
        print(f"FID with dataloaders: {fid_dataloaders:0.3g} ({elapsed_time/fid.get_last_num_imgs()*1000:.3f} s/1000 images)")

        print("Reading all images")
        train_A_imgs = torch.empty(0)
        for batch in tqdm(self.train_A):
            train_A_imgs = torch.concat([train_A_imgs, batch])
        train_B_turbo_imgs = torch.empty(0)
        for batch in tqdm(self.train_B_turbo):
            train_B_turbo_imgs = torch.concat([train_B_turbo_imgs, batch])

        print("FID with images")
        start_time = time.time()
        fid_imgs = fid.get(train_A_imgs, train_B_turbo_imgs)
        elapsed_time = time.time() - start_time
        print(f"FID with images: {fid_imgs:0.3g} ({elapsed_time/fid.get_last_num_imgs()*1000:.3f} s/1000 images)")

        self.assertAlmostEqual(fid_dataloaders, fid_imgs, 3, "Values shoud be very close.")

    def test_lpips_dataloader(self):
        """Test LPIPS calculation using dataloaders and images."""
        print("=========================")
        print("LPIPS calculation options")
        print("=========================")

        if self.print_memory:
            print(get_gpu_memory_usage("Initital memory usage", short_msg=True))

        lpips = LPIPS(cuda=self.use_cuda, batch_size=16)
        if self.print_memory:
            print(get_gpu_memory_usage("After model load", short_msg=True))

        print("LPIPS with data loaders")
        n = min(len(self.train_A.dataset), len(self.train_B_turbo.dataset))
        data1_ = copy_dataloader(self.train_A)
        data2_ = copy_dataloader(self.train_B_turbo)
        data1_.dataset.set_len(n)
        data2_.dataset.set_len(n)

        start_time = time.time()
        lpips_dataloaders = lpips.get(data1_, data2_)
        elapsed_time = time.time() - start_time
        print(f"LPIPS with dataloaders: {lpips_dataloaders.mean():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 images)")

        print("Reading all images")
        train_A_imgs = torch.empty(0)
        for batch in tqdm(self.train_A):
            train_A_imgs = torch.concat([train_A_imgs, batch])
        train_B_turbo_imgs = torch.empty(0)
        for batch in tqdm(self.train_B_turbo):
            train_B_turbo_imgs = torch.concat([train_B_turbo_imgs, batch])

        print("LPIPS with images")
        start_time = time.time()
        lpips_imgs = lpips.get(train_A_imgs[:n], train_B_turbo_imgs[:n])
        elapsed_time = time.time() - start_time
        print(f"LPIPS with images: {lpips_imgs.mean():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 images)")

        self.assertAlmostEqual(
            float(lpips_dataloaders.mean()),
            float(lpips_imgs.mean()),
            2,
            "Values shoud be close."
        )

    def test_fid(self):
        """Test FID calculation."""
        print("===============")
        print("FID calculation")
        print("===============")
        if self.print_memory:
            print(get_gpu_memory_usage("Initital memory usage", short_msg=True))

        fid = FID(dims=2048, cuda=self.use_cuda)
        if self.print_memory:
            print(get_gpu_memory_usage("After model load", short_msg=True))

        start_time = time.time()
        fid_equal = fid.get(self.train_A, self.train_A)
        elapsed_time = time.time() - start_time
        print(f"FID score of the same images: {fid_equal:0.3g} ({elapsed_time/fid.get_last_num_imgs()*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_same_A = fid.get(self.train_A, self.test_A)
        elapsed_time = time.time() - start_time
        print(f"FID score of only A images: {fid_same_A:0.3g} ({elapsed_time/fid.get_last_num_imgs()*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_same_B = fid.get(self.train_B, self.test_B)
        elapsed_time = time.time() - start_time
        print(f"FID score of only B images: {fid_same_B:0.3g} ({elapsed_time/fid.get_last_num_imgs()*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_different = fid.get(self.train_A, self.train_B)
        elapsed_time = time.time() - start_time
        print(f"FID score of A x B images: {fid_different:0.3g} ({elapsed_time/fid.get_last_num_imgs()*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_turbo_B = fid.get(self.test_A, self.train_B_turbo)
        elapsed_time = time.time() - start_time
        print(f"FID score of A images and CycleGAN-turbo transformed images: {fid_turbo_B:0.3g} ({elapsed_time/fid.get_last_num_imgs()*1000:.3f} s/1000 images)")

        start_time = time.time()
        fid_turbo_A = fid.get(self.test_B, self.train_A_turbo)
        elapsed_time = time.time() - start_time
        print(f"FID score of B images and CycleGAN-turbo transformed images: {fid_turbo_A:0.3g} ({elapsed_time/fid.get_last_num_imgs()*1000:.3f} s/1000 images)")

        if self.print_memory:
            print(get_gpu_memory_usage("After FID calculation", short_msg=True))
        torch.cuda.empty_cache()
        if self.print_memory:
            print(get_gpu_memory_usage("After emptying cache", short_msg=True))


        self.assertLess(fid_equal, 1E-3, "FID for same images should be zero.")
        self.assertLess(fid_same_A, fid_different, "FID for images of the same class A should be lower than for images of different classes.")
        self.assertLess(fid_same_B, fid_different, "FID for images of the same class B should be lower than for images of different classes.")


    def test_lpips(self):
        """Test LPIPS calculation."""
        print("=================")
        print("LPIPS calculation")
        print("=================")
        if self.print_memory:
            print(get_gpu_memory_usage("Initital memory usage", short_msg=True))

        lpips = LPIPS(cuda=self.use_cuda)
        if self.print_memory:
            print(get_gpu_memory_usage("After model load", short_msg=True))

        start_time = time.time()
        lpips_train_a = lpips.get(self.train_A, self.train_A)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of train A images: {lpips_train_a.mean():0.3g} ± {lpips_train_a.std():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 image pairs)")

        start_time = time.time()
        lpips_train_a_pairs = lpips.get(self.train_A, self.train_A, all_pairs=True)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of train A images with more pairs: {lpips_train_a_pairs.mean():0.3g} ± {lpips_train_a_pairs.std():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 image pairs)")


        start_time = time.time()
        lpips_test_a = lpips.get(self.test_A, self.test_A)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of test A images: {lpips_test_a.mean():0.3g} ± {lpips_test_a.std():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 image pairs)")

        start_time = time.time()
        lpips_test_a_pairs = lpips.get(self.test_A, self.test_A, all_pairs=True)
        elapsed_time = time.time() - start_time
        print(f"LPIPS loss of test A images with more pairs: {lpips_test_a_pairs.mean():0.3g} ± {lpips_test_a_pairs.std():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 image pairs)")


        def _make_a_test(data1, data2, all_pairs=False):
            n = min(len(data1.dataset), len(data2.dataset))
            data1_ = copy_dataloader(data1)
            data2_ = copy_dataloader(data2)
            data1_.dataset.set_len(n)
            data2_.dataset.set_len(n)
            start_time = time.time()
            lpips_value = lpips.get(data1_, data2_, all_pairs=all_pairs)
            elapsed_time = time.time() - start_time
            return lpips_value, elapsed_time

        lpips_same_a, elapsed_time = _make_a_test(self.train_A, self.test_A, all_pairs=False)
        print(f"LPIPS loss of train and test A images: {lpips_same_a.mean():0.3g} ± {lpips_same_a.std():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 image pairs)")

        lpips_same_b, elapsed_time = _make_a_test(self.train_B, self.test_B, all_pairs=False)
        print(f"LPIPS loss of train and test B images: {lpips_same_b.mean():0.3g} ± {lpips_same_b.std():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 image pairs)")

        lpips_different, elapsed_time = _make_a_test(self.train_A, self.train_B, all_pairs=False)
        print(f"LPIPS loss of train A and B images: {lpips_different.mean():0.3g} ± {lpips_different.std():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 image pairs)")

        lpips_turbo_a, elapsed_time = _make_a_test(self.train_A, self.train_B_turbo, all_pairs=False)
        print(f"LPIPS loss of train A and CycleGAN-turbo images: {lpips_turbo_a.mean():0.3g} ± {lpips_turbo_a.std():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 image pairs)")

        lpips_turbo_b, elapsed_time = _make_a_test(self.train_B, self.train_A_turbo, all_pairs=False)
        print(f"LPIPS loss of train B and CycleGAN-turbo images: {lpips_turbo_b.mean():0.3g} ± {lpips_turbo_b.std():0.3g} ({elapsed_time/lpips.get_last_num_pairs()*1000:.3f} s/1000 image pairs)")

        if self.print_memory:
            print(get_gpu_memory_usage("After LPIPS calculation", short_msg=True))
        torch.cuda.empty_cache()
        if self.print_memory:
            print(get_gpu_memory_usage("After emptying cache", short_msg=True))

        self.assertAlmostEqual(
            float(lpips_train_a_pairs.mean()),
            float(lpips_train_a.mean()),
            2,
            "Train A: LPIPS loss with additional pairs should get approximately same result."
        )
        self.assertAlmostEqual(
            float(lpips_test_a_pairs.mean()),
            float(lpips_test_a.mean()),
            2,
            "Test A: LPIPS loss with additional pairs should get approximately same result."
        )
        self.assertLess(lpips_same_a.mean(), lpips_different.mean(), "LPIPS loss for images of the same A class should be lower than for images of different classes.")
        self.assertLess(lpips_same_b.mean(), lpips_different.mean(), "LPIPS loss for images of the same B class should be lower than for images of different classes.")

        self.assertLess(lpips_same_a.mean(), lpips_turbo_a.mean(), "LPIPS loss for images of the same A class should be lower than for images generated by CycleGAN turbo.")
        self.assertLess(lpips_same_b.mean(), lpips_turbo_a.mean(), "LPIPS loss for images of the same B class should be lower than for images generated by CycleGAN turbo.")

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestMetrics('test_fid_dataloader'))
    suite.addTest(TestMetrics('test_lpips_dataloader'))
    suite.addTest(TestMetrics('test_fid'))
    suite.addTest(TestMetrics('test_lpips'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
