# pylint: disable=line-too-long
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
    requirements_file = str(requirements_file)
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
            with Image.open(src_path) as img:
                img.save(dst_path)
        else:
            print(f"File not found: {src_path.name}")

def search_and_copy_b_images(df, src_dir, base_img_dir, dst_dir, threshold=1e-5):
    """
    Search similar images in the source directory and copy them to the destination directory.

    Parameters:
    ------------
    df: pd.DataFrame
        The DataFrame containing the image file names.
    src_dir: str or Path
        The path to the source directory containing the images to be copied.
    base_img_dir: str or Path
        The path to the base directory containing the images to be compared with.
    dst_dir: str or Path
        The path to the destination directory where the images will be copied.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    all_images = []
    print("Loading images...")
    for file_path in tqdm(src_dir.rglob('*_real_B.png')):
        img = np.array(Image.open(file_path).convert('RGB'))
        all_images.append((file_path,img))

    print("Comparing images...")
    used = []
    for _, row in tqdm(df.iterrows()):
        src_path = Path(base_img_dir) / row['file_name']
        src_image = Image.open(src_path).convert('RGB')
        src_image = np.array(src_image)

        i = len(used)
        for file_path, img in all_images:
            if file_path in used:
                continue
            if np.max(np.abs(src_image - img) < threshold):
                used.append(file_path)
                new_name = file_path.stem.replace('_real_B','_fake_A') + file_path.suffix
                file_path = file_path.with_name(new_name)
                with Image.open(file_path) as img:
                    img.save(dst_dir / src_path.name)
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

    print("Processing A images")
    df_imgs_A_train = pd.read_csv(base_out_folder / 'input_A_train.csv')
    df_imgs_A_test = pd.read_csv(base_out_folder / 'input_A_test.csv')
    copy_images(
        df=pd.concat([df_imgs_A_train, df_imgs_A_test]),
        src_dir=base_src_folder,
        dst_dir=base_out_folder / 'output_A_cyclegan',
        src_name_modifier='_fake_B')

    # Images B
    print("Processing B images")
    df_imgs_B_train = pd.read_csv(base_out_folder / 'input_B_train.csv')
    df_imgs_B_test = pd.read_csv(base_out_folder / 'input_B_test.csv')
    search_and_copy_b_images(
        df=pd.concat([df_imgs_B_train, df_imgs_B_test]),
        src_dir=base_src_folder,
        base_img_dir=base_out_folder / 'input_B',
        dst_dir=base_out_folder / 'output_B_cyclegan',
        threshold=5,
    )
