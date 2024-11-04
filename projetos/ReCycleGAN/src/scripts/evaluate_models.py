# pylint: disable=import-error,wrong-import-position,line-too-long
"""Script to load and evaluate models."""
import sys
import itertools
from pathlib import Path
import pickle
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
from src.metrics.lpips import LPIPS


def create_2d_map(distances):
    """Create a 2D map of points given a list of distances between the points."""
    distances = np.array(distances)
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
        plt.text(df['value'].max()*0.025, index, f'{value:.4g}', color='black', ha="left", va="center")

    if title is not None:
        plt.title(title, fontsize=16, weight='bold')
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_stacked_histograms(tensors, labels, file_path, bins=10, title=None, xlabel='Value', figsize=(8, 2)):
    """Plot stacked histograms."""
    num_plots = len(tensors)
    _, axes = plt.subplots(num_plots, 1, figsize=(figsize[0], figsize[1] * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    for i, (tensor, ax) in enumerate(zip(tensors, axes)):
        array = tensor.numpy()
        sns.histplot(array, bins=bins, kde=False, ax=ax, edgecolor='black',stat='density', alpha=0.5)
        ax.set_ylabel(labels[i])
        ax.grid(False)

    sns.despine()
    axes[-1].set_xlabel(xlabel)
    if title is not None:
        plt.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def build_images(case, device='cuda'):
    """Generates translated images the CycleGAN model."""
    print(f"Building Translated Images for Test Case {case}")
    params = TEST_CASES[str(case)]

    restart_folder = BASE_FOLDER / f'data/checkpoints/test_case_{case}'
    restart_file = list(restart_folder.glob('*.pth'))
    if len(restart_file) == 0:
        print(f"No pth file not found in: {restart_folder}")
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
            images_csv = BASE_FOLDER / f'data/external/nexet/input_{p}_all_filtered.csv'
            f_name = f'input_{p}'
        else:
            fake_p = 'B' if p == 'A' else 'A'
            images_csv = BASE_FOLDER / f'data/external/nexet/input_{fake_p}_all_filtered.csv'
            if folder_name == 'oposite':
                f_name = f'input_{fake_p}'
            else:
                f_name = f'output_{fake_p}_{folder_name}'
        out[p] = get_img_dataloader(
            csv_file = images_csv,
            img_dir = BASE_FOLDER / f'data/external/nexet/{f_name}',
            batch_size = 128,
            transformation = None
        )
    return out


def get_fid(data_loaders, use_cuda=True):
    """Calculates the FID score for a list of models."""
    print('Loading FID model')
    fid = FID(dims=2048, cuda=use_cuda)

    statistics = {}
    for k,v in data_loaders.items():
        print(f"Calculating features statistics for {k}")
        statistics[k] = {}
        for p,imgs in v.items():
            statistics[k][p] = fid.compute_statistics_of_imgs(imgs)

    pairs = list(itertools.combinations(data_loaders.keys(), 2))

    print(f'Calculating FID for all {len(pairs)} pairs')
    results = {'A':{}, 'B':{}}
    for pair in tqdm(pairs):
        for p in ['A','B']:
            m1, s1 = statistics[pair[0]][p]
            m2, s2 = statistics[pair[1]][p]
            results[p][pair] = fid.calculate_frechet_distance(m1, s1, m2, s2)
    return results


def get_lpips(data_loaders, use_cuda=True):
    """Calculates the LPIPS score for a list of models."""
    print('Loading LPIPS model')
    lpips = LPIPS(cuda=use_cuda)

    pairs = list(itertools.combinations(data_loaders.keys(), 2))
    print(f'Calculating LPIPS for all {len(pairs)} pairs')
    results = {'A':{}, 'B':{}}
    for pair in pairs:
        for p in ['A','B']:
            imgs1 = data_loaders[pair[0]][p]
            imgs2 = data_loaders[pair[1]][p]
            n = min(len(imgs1.dataset), len(imgs2.dataset))
            imgs1.dataset.set_len(n)
            imgs2.dataset.set_len(n)
            results[p][pair] = lpips.lpips_dataloader(
                imgs1, imgs2, description=str(pair),
                normalize=False, use_all_pairs=False)
    return results


def distance_dict_to_table(distances, keys):
    """Transform dict of distances into table."""
    d_table = np.zeros([len(keys),len(keys)])
    for i, k1 in enumerate(keys[:-1]):
        for j, k2 in enumerate(keys[i+1:]):
            d_table[i,j+i+1] = distances[(k1,k2)]
            d_table[j+i+1,i] = distances[(k1,k2)]
    return d_table


def transform_distances(distances, transform):
    """Transform distances."""
    out = {}
    for p in ['A','B']:
        out[p] = {}
        for k,v in distances[p].items():
            out[p][k] = transform(v)
    return out


def print_distance_pairs(distances):
    """Print distance pairs."""
    for p in ['A','B']:
        print(f"Distances for {p} images")
        for k,v in distances[p].items():
            print(f"\t{k[0]} - {k[1]}: {v:.4g}")
        print()


def plot_distances(distances, labels, title):
    """Plot distances."""
    for p in ['A','B']:
        table = distance_dict_to_table(distances[p], labels)
        points_2d = create_2d_map(table)
        plot_2d_map(
            points_2d,
            labels,
            file_path=BASE_FOLDER / f'docs/assets/evaluation/{title.lower()}_map_images_{p}.png',
            title=f'{title.upper()} for {p} Images')

        data = {
            'class': labels[1:],
            'value': [distances[p][(labels[0],k)] for k in labels[1:]]
        }
        plot_hbar(
            data,
            file_path=BASE_FOLDER / f'docs/assets/evaluation/{title.lower()}_bar_images_{p}.png',
            title=f'{title.upper()} for {p} Images',
            x_label=f'{title.upper()} to Real Images',
            y_label='')


def plot_histograms(distances, labels, title):
    """Plot histograms."""
    for p in ['A','B']:
        distances_list = []
        for k in labels[1:]:
            distances_list.append(distances[p][('Real',k)].flatten())
        plot_stacked_histograms(
            distances_list,
            labels[1:],
            file_path=BASE_FOLDER / f'docs/assets/evaluation/{title.lower()}_histograms_{p}.png',
            title=f'{title.upper()} for {p} Images', xlabel=f'{title.upper()} to Real Images',
            figsize=(8, 1.5))


def save_distances(distances, file_name):
    """Save distances to file."""
    file_path = BASE_FOLDER / f'docs/assets/evaluation/{file_name}'
    with open(file_path, 'wb') as file:
        pickle.dump(distances, file)

def load_distances(file_name):
    """Load distances from file."""
    file_path = BASE_FOLDER / f'docs/assets/evaluation/{file_name}'
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def main():
    """Main function."""

    # Build translated images
    # for i_ in range(1, 5):
    #     build_images(i_)

    # Build image data loaders
    model_list = {
        'Real': 'real',
        'Oposite class': 'oposite',
        'CycleGAN': 'cyclegan',
        'CycleGAN-turbo': 'turbo',
    }
    for i in range(1, 5):
        test_case = TEST_CASES[str(i)]
        model_list[test_case['short_description']] = f'test_{i}'

    data_loaders = {}
    for k,v in model_list.items():
        data_loaders[k] = build_data_loaders(v)

    fid_distances = get_fid(data_loaders)
    print_distance_pairs(fid_distances)
    labels = list(model_list.keys())
    plot_distances(fid_distances, labels, 'FID')
    save_distances(fid_distances, 'fid_distances.pkl')

    lpips_distances = get_lpips(data_loaders)
    lpips_distances_mean = transform_distances(lpips_distances, transform=lambda x: float(x.mean()))
    print_distance_pairs(lpips_distances_mean)
    labels = list(model_list.keys())
    plot_distances(lpips_distances_mean, labels, 'LPIPS')
    plot_histograms(lpips_distances, labels, 'LPIPS')
    save_distances(lpips_distances, 'lpips_distances.pkl')

    # Calculate LPIPS
    #   Sample images along histogram
    # Build samples with all translations

if __name__ == '__main__':
    main()
