# # Extract DayNight Images
# 
# Create DayNight database for model training.

import sys
from pathlib import Path
import pandas as pd

current_dir = Path().resolve()
sys.path.append(str(current_dir.parent))
import utils

labels_file_name = '../../data/external/nexet/nexet/train.csv'

df_labels = pd.read_csv(labels_file_name)

img_folder = '../../data/external/nexet/nexet/nexet_2017_1/'

img_folder = Path(img_folder)
file_count = len([f for f in img_folder.iterdir() if f.is_file()])
print(f'There are {file_count} files in the folder.')

df_count = utils.img_size_count(img_folder)
utils.img_size_count_plot(df_count, show=True)

folders = {
        'input_A': {'lighting':['Day'], 'city':['NYC']},
        'input_B': {'lighting':['Night'], 'city':['NYC']},
    }

for out_folder, df_filter in folders.items():
    img_list = utils.filter_dataframe(df_labels, df_filter)['image_filename']
    utils.Images.build_dataset(
        img_list=img_list,
        img_folder=Path(img_folder),
        output_folder=Path('../../data/external/nexet/') / out_folder,
        transformation_function=utils.resize_and_crop,
        transformation_params={'target_size':(256,256), 'size_filter':[(1280, 720)]},
        split=0.8,
        random_seed=42
    )


