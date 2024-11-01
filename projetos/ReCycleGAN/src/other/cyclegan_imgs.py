"""Generate day to night images using the CycleGAN model.

A new Python virtual environment is recommended to run this script.
The path to img2img-turbo source code must be adjusted as needed.
"""

import os
import subprocess
from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np

def install_requirements(requirements_file):
    """
    Install packages from a requirements.txt file using pip.

    Parameters:
    ------------
    requirements_file: str or Path
        The path to the requirements.txt file.
    """
    requirements_file = str(requirements_file)  # Ensure the path is a string
    try:
        subprocess.check_call(['pip', 'install', '-r', requirements_file])
    except subprocess.CalledProcessError as e:
        msg = f'Failed to install packages from {requirements_file}'
        raise RuntimeError(msg) from e

def run(command):
    """Run a shell command."""
    try:
        subprocess.check_call(command.split())
    except subprocess.CalledProcessError as e:
        msg = 'Failed to train CycleGAN'
        raise RuntimeError(msg) from e

def copy_images(df, src_dir, dst_dir, src_name_modifier=''):
    """Copy images from the source directory to the destination directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(df.iterrows()):
        src_path = Path(src_dir) / row['file_name']
        src_path = src_path.with_name(src_path.stem + src_name_modifier + src_path.suffix)
        src_path = src_path.with_suffix('.png')
        dst_path = Path(dst_dir) / row['file_name']
        if src_path.is_file():
            shutil.copy(src_path, dst_path)
        else:
            print(f"File not found: {src_path.name}")


def compare_images(image1, image_path2, threshold=1e-5):
    """
    Compare two images to check if they are the same.

    Parameters:
    ------------
    image_path1: str or Path
        The path to the first image file.
    image_path2: str or Path
        The path to the second image file.
    threshold: float
        The maximum allowed absolute difference between the images.

    Returns:
    ------------
    bool
        True if the images are the same, False otherwise.
    """
    # image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')

    # Resize images to the same size if they are different
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    # Convert images to numpy arrays
    np_image1 = np.array(image1)
    np_image2 = np.array(image2)

     # Compute the maximum absolute difference between the images
    max_abs_diff = np.max(np.abs(np_image1 - np_image2))

    # Compare the maximum absolute difference with the threshold
    return max_abs_diff <= threshold

def search_and_copy_images(df, src_dir, base_img_dir, dst_dir, threshold=1e-5):
    """
    Search similar images in the source directory and copy them to the destination directory.

    Parameters:
    ------------
    df: pd.DataFrame
        The DataFrame containing the image file names.
    src_dir: str or Path
        The path to the source directory containing the images to be copied.
    base_img_dir: str or Path
        The path to the base directory containing the images to be compared.
    dst_dir: str or Path
        The path to the destination directory where the images will be copied.
    """
    used = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(df.iterrows()):
        src_path = Path(base_img_dir) / row['file_name']
        src_image = Image.open(src_path).convert('RGB')

        i = len(used)
        for file_path in src_dir.rglob('*_real_B.png'):
            if file_path in used:
                continue
            if file_path.is_file():
                if compare_images(src_image, file_path, threshold=threshold):
                    file_path = file_path.with_name(file_path.stem.replace('_real_B','_fake_A') + file_path.suffix)
                    shutil.copy(file_path, dst_dir / src_path.name)
                    used.append(file_path)
                    break
        if i == len(used):
            print(f"Similar image not found for: {src_path.name}")

if __name__ == '__main__':

    cyclegan_path = Path(__file__).parent.parent.parent.parent.parent.parent / 'pytorch-CycleGAN-and-pix2pix'
    if not cyclegan_path.exists():
        print(f"Directory not found: {cyclegan_path}")
        raise FileNotFoundError
    os.chdir(cyclegan_path)

    # install_requirements(cyclegan_path / 'requirements.txt')

    # train model
    # train_command = '!python train.py --dataroot ./datasets/nexet --name day2night --model cycle_gan --display_id -1'
    # run(train_command)

    # generate test images
    # test_command = '!python test.py --dataroot datasets/nexet/ --name day2night --model cycle_gan --num_test 999999'
    # run(test_command)

    base_src_folder = cyclegan_path / 'results/day2night/test_latest/images'
    base_out_folder = Path(__file__).parent.parent.parent / 'data/external/nexet'

    # Images A
    for df_name in ['input_A_train.csv', 'input_A_test.csv']:
        print(f"Processing {df_name}")
        df_imgs = pd.read_csv(base_out_folder / df_name)
        copy_images(
            df=df_imgs,
            src_dir=base_src_folder,
            dst_dir=base_out_folder / 'output_A_cyclegan',
            src_name_modifier='_fake_B')

    # Images B
    for df_name in ['input_B_train.csv', 'input_B_test.csv']:
        print(f"Processing {df_name}")
        df_imgs = pd.read_csv(base_out_folder / df_name)
        search_and_copy_images(
            df=df_imgs,
            src_dir=base_src_folder,
            base_img_dir=base_out_folder / 'input_B',
            dst_dir=base_out_folder / 'output_B_cyclegan',
            threshold=5,
        )
