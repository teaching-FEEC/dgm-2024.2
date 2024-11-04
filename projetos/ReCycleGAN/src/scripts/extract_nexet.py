# pylint: disable=import-error,wrong-import-position
"""Create DayNight database for model training."""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import ImageTools, utils

df_labels = pd.read_csv('projetos/ReCycleGAN/no_sync/nexet/nexet/train.csv')
img_folder = Path('projetos/ReCycleGAN/no_sync/nexet/nexet/nexet_2017_1/')
out_folder = Path('projetos/ReCycleGAN/data/external/nexet/')

file_count = len([f for f in img_folder.iterdir() if f.is_file()])
print(f'There are {file_count} files in the folder.')

df_count = ImageTools.img_size_count(img_folder)
ImageTools.img_size_count_plot(df_count)
out_folder.mkdir(parents=True, exist_ok=True)
plt.savefig(out_folder / 'img_size_count.png')
plt.close()

folders = {
        'input_A': {'lighting':['Day'], 'city':['NYC']},
        'input_B': {'lighting':['Night'], 'city':['NYC']},
    }

for folder, df_filter in folders.items():
    img_list = utils.filter_dataframe(df_labels, df_filter)['image_filename']
    ImageTools.build_dataset(
        img_list=img_list,
        img_folder=Path(img_folder),
        output_folder=out_folder / folder,
        transformation_function=ImageTools.resize_and_crop,
        transformation_params={'target_size':(256,256), 'size_filter':[(1280, 720)]},
        split=0.8,
        random_seed=42
    )
