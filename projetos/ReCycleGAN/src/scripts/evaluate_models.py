# pylint: disable=import-error,wrong-import-position,line-too-long
"""Script to load and evaluate models."""
import sys
import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from test_model import translate_images

BASE_FOLDER = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_FOLDER))
from src.utils.test_cases import TEST_CASES
from src.utils.utils import save_dict_as_json
from src.utils.data_loader import get_img_dataloader
from src.metrics.fid import FID
# from src.metrics.lpips import LPIPS

def build_images(case, device='cuda'):
    """Generates translated images the CycleGAN model."""
    print(f"Test Case {case}")
    params = TEST_CASES[str(case)]

    restart_folder = BASE_FOLDER / f'data/checkpoints/test_case_{case}'
    restart_file = list(restart_folder.glob('*.pth'))
    if len(restart_file) == 0:
        print(f"Np pth file not found in: {restart_folder}")
        return
    if len(restart_file) > 1:
        print(f"Multiple pth files found in: {restart_folder}")
        return

    params['restart_path'] = restart_file[0]
    params['data_folder'] = BASE_FOLDER / 'data/external/nexet'
    params['output_name'] = f'test_{case}'
    params['csv_type'] = ''
    params['device'] = device
    save_dict_as_json(params, restart_file[0].with_suffix('.json'))
    translate_images(params)


def build_data_loaders(folder_name):
    """Builds the data loaders for the images."""
    out = {}
    for p in ['A','B']:
        if folder_name == 'real':
            f_name = f'input_{p}'
            images_csv = BASE_FOLDER / f'data/external/nexet/input_{p}_all_filtered.csv'
        else:
            if p == 'A':
                fake_p = 'B'
            else:
                fake_p = 'A'
            f_name = f'output_{fake_p}_{folder_name}'
            images_csv = BASE_FOLDER / f'data/external/nexet/input_{fake_p}_all_filtered.csv'
        out[p] = get_img_dataloader(
            csv_file = images_csv,
            img_dir = BASE_FOLDER / f'data/external/nexet/{f_name}',
            batch_size = 128,
            transformation = None
        )
    return out

def get_fid(data_loaders, use_cuda=True):
    """Calculates the FID score for a list of models."""
    fid = FID(dims=2048, cuda=use_cuda)

    statistics = {}
    for k,v in data_loaders.items():
        print(f"Calculating features statistics for {k}")
        statistics[k] = {}
        for p,imgs in v.items():
            statistics[k][p] = fid.compute_statistics_of_imgs(imgs)

    pairs = list(itertools.combinations(data_loaders.keys(), 2))

    print('Calculating FID for all pairs')
    results = {'A':{}, 'B':{}}
    for pair in tqdm(pairs):
        for p in ['A','B']:
            m1, s1 = statistics[pair[0]][p]
            m2, s2 = statistics[pair[1]][p]
            results[p][pair] = fid.calculate_frechet_distance(m1, s1, m2, s2)

    return results

def create_2d_map(distances):
    """
    Create a 2D map of points given a list of distances between the points.

    Parameters:
    ------------
    distances: list of list of float
        A 2D list representing the distances between points.

    Returns:
    ------------
    np.ndarray
        A 2D array representing the coordinates of the points.
    """
    # Convert the list of distances to a numpy array
    distances = np.array(distances)

    # Perform MDS to find the 2D coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    points_2d = mds.fit_transform(distances)

    return points_2d

def plot_2d_map(points_2d, labels, file_path, title=None, label_dist=0.025):
    """Plot a 2D map of points."""

    df = pd.DataFrame(points_2d, columns=['x', 'y'])
    df['label'] = labels
    plt.figure(figsize=(6, 4))
    scatter = sns.scatterplot(data=df, x='x', y='y', hue='label', palette='viridis', s=100)
    plt.gca().set_aspect('equal', adjustable='box')

    if title is not None:
        plt.title(title, fontsize=16, weight='bold')
    scatter.legend_.remove()

    plt.xlabel('')
    plt.ylabel('')
    # plt.gca().set_xticklabels([])
    # plt.gca().set_yticklabels([])
    plt.grid(True)

    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range > y_range:
        y_min_ = (y_min+y_max)/2.0 - x_range / 2.0
        y_max_ = (y_min+y_max)/2.0 + x_range / 2.0
        plt.gca().set_ylim(y_min_, y_max_)
    elif y_range > x_range:
        x_min_ = (x_min+x_max)/2.0 - y_range / 2.0
        x_max_ = (x_min+x_max)/2.0 + y_range / 2.0
        plt.gca().set_xlim(x_min_, x_max_)

    y_min, y_max = plt.gca().get_ylim()
    y_range = y_max - y_min
    label_dist *= y_range
    for i in range(len(df)):
        plt.text(df['x'][i], df['y'][i] + label_dist, df['label'][i],
                 horizontalalignment='center',
                 size='small', #  fontsize=9,
                 color='black',
        )

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_hbar(data, file_path, title=None, x_label=None, y_label=None):
    """Plot horizontal bars."""
    df = pd.DataFrame(data)
    plt.figure(figsize=(6, 4))
    sns.barplot(y='class', x='value', data=df, edgecolor='gray', color='skyblue')
    for index, value in enumerate(df['value']):
        plt.text(df['value'].max()*0.025, index, f'{value:.2f}', color='black', ha="left", va="center")

    if title is not None:
        plt.title(title, fontsize=16, weight='bold')
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def distance_dict_to_table(distances, keys):
    """Transform dict of distances to table."""
    d_table = np.zeros([len(keys),len(keys)])
    for i, k1 in enumerate(keys[:-1]):
        for j, k2 in enumerate(keys[i+1:]):
            d_table[i,j+i+1] = distances[(k1,k2)]
            d_table[j+i+1,i] = distances[(k1,k2)]
    return d_table

if __name__ == '__main__':

    # Build translated images
    # for i_ in range(1, 4):
    #     build_images(i_)

    # Build image data loaders
    model_list_ = {
        'Real': 'real',
        'CycleGAN': 'cyclegan',
        'CycleGAN-turbo': 'turbo',
    }
    for i_ in range(1, 4):
        model_list_[TEST_CASES[str(i_)]['short_description']] = f'test_{i_}'

    data_loaders_ = {}
    for k_,v_ in model_list_.items():
        data_loaders_[k_] = build_data_loaders(v_)

    # Calculate FID distances
    fid_dict_ = get_fid(data_loaders_)

    print('FID distances')
    for p_ in ['A','B']:
        print(f'Imgs {p_}')
        for k_,v_ in fid_dict_[p_].items():
            print(f'\t{str(k_)}: {v_:0.3f}')

    labels_ = list(model_list_.keys())
    for p_ in ['A','B']:
        fid_table_ = distance_dict_to_table(fid_dict_[p_], labels_)
        fid_2d_points_ = create_2d_map(fid_table_)
        plot_2d_map(
            fid_2d_points_,
            labels_,
            file_path=BASE_FOLDER / f'docs/assets/evaluation/fid_map_images_{p_}.png',
            title=f'FID for {p_} Images')

        data_ = {
            'class': labels_[1:],
            'value': [fid_dict_[p_][(labels_[0],k)] for k in labels_[1:]]
        }
        plot_hbar(
            data_,
            file_path=BASE_FOLDER / f'docs/assets/evaluation/fid_bar_images_{p_}.png',
            title=f'FID for {p_} Images',
            x_label='FID to Real Images')
