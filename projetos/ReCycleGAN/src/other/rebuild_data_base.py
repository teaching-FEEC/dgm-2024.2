"""Rebuild the image database using CycleGAN format."""

from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm

def copy_images(df, src_dir, dst_dir):
    """Copy images from the source directory to the destination directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(df.iterrows()):
        src_path = Path(src_dir) / row['file_name']
        dst_path = Path(dst_dir) / row['file_name']
        if src_path.is_file():
            shutil.copy(src_path, dst_path)
        else:
            print(f"File not found: {src_path.name}")

if __name__ == '__main__':
    base_src_folder = Path(__file__).parent.parent.parent / 'data/external/nexet'
    base_out_folder = Path(__file__).parent.parent.parent / 'data/external/nexet/nexet'

    df_list = [
        'input_A_train_filtered.csv',
        'input_B_train_filtered.csv',
        'input_A_test_filtered.csv',
        'input_B_test_filtered.csv'
    ]
    src_folder = ['input_A', 'input_B', 'input_A', 'input_B']
    out_folder = ['trainA', 'trainB', 'testA', 'testB']

    for df_name, src, out in zip(df_list, src_folder, out_folder):
        df_imgs = pd.read_csv(base_src_folder / df_name)
        copy_images(df_imgs, base_src_folder / src, base_out_folder / out)

    shutil.make_archive(base_out_folder, 'zip', base_out_folder)
    shutil.rmtree(base_out_folder)


#Code to run in the Colab CycleGAN notebook
# https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb

# import zipfile
# with zipfile.ZipFile('/content/nexet.zip', 'r') as zip_ref:
#     zip_ref.extractall('/content/pytorch-CycleGAN-and-pix2pix/datasets/nexet')
