# pylint: disable=import-error,wrong-import-position,line-too-long
"""Script to load and evaluate models."""
import sys
import itertools
from pathlib import Path
import pickle
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
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
from src.utils.data_loader import get_img_dataloader, copy_dataloader
from src.metrics.fid import FID
from src.metrics.lpips import LPIPS
from src.utils.data_transform import ImageTools

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

    if label_dist > 0:
        plt.figure(figsize=(6, 4))
    else:
        plt.figure(figsize=(6.5, 4))

    scatter = sns.scatterplot(data=df, x='x', y='y', hue='label', palette='Set1', s=100)
    plt.gca().set_aspect('equal', adjustable='box')

    if title is not None:
        plt.title(title, fontsize=16, weight='bold')
    if label_dist > 0:
        scatter.legend_.remove()
    else:
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

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

    if label_dist > 0:
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
    if 'std' in df.columns:
        sns.barplot(y='class', x='value', data=df, edgecolor='gray', color='skyblue', xerr=df['std'])
    else:
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


def plot_stacked_histograms(tensors, labels, file_path, bins=20, title=None, xlabel='Value', figsize=(8, 2)):
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
    """Calculates the FID score for all pairs in a list of models."""
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
    """Calculates the LPIPS score for all pairs in a list of models."""
    print('Loading LPIPS model')
    lpips = LPIPS(cuda=use_cuda)

    pairs = list(itertools.combinations(data_loaders.keys(), 2))
    print(f'Calculating LPIPS for all {len(pairs)} pairs')
    results = {'A':{}, 'B':{}}
    for p in ['A','B']:
        for pair in pairs + [('Real','Real')]:
            imgs1 = copy_dataloader(data_loaders[pair[0]][p])
            imgs2 = copy_dataloader(data_loaders[pair[1]][p])
            n = min(len(imgs1.dataset), len(imgs2.dataset))
            imgs1.dataset.set_len(n)
            imgs2.dataset.set_len(n)
            results[p][pair] = lpips.lpips_dataloader(
                imgs1, imgs2, description=str(pair),
                normalize=False, use_all_pairs=False)
    return results

def lpips_distance(mu, sigma):
    """Calculate Wasserstein distances from LPIPS statistics."""
    out = {'A':{}, 'B':{}}
    for p in ['A','B']:
        mu2 = mu[p][('Real','Real')]
        sigma2 = sigma[p][('Real','Real')]
        for k in mu[p]:
            if k[0] != k[1]:
                mu1 = mu[p][k]
                sigma1 = sigma[p][k]
                out[p][k] = np.sqrt((mu1-mu2)**2 + (sigma1-sigma2)**2)
    return out

def metric_dict_to_table(metrics, keys):
    """Transform dict of metrics into table."""
    d_table = np.zeros([len(keys),len(keys)])
    for i, k1 in enumerate(keys[:-1]):
        for j, k2 in enumerate(keys[i+1:]):
            d_table[i,j+i+1] = metrics[(k1,k2)]
            d_table[j+i+1,i] = metrics[(k1,k2)]
    return d_table


def transform_metrics(metrics, transform):
    """Transform metrics."""
    out = {}
    for p in ['A','B']:
        out[p] = {}
        for k,v in metrics[p].items():
            out[p][k] = transform(v)
    return out


def print_metric_pairs(metrics1, metrics2=None):
    """Print metric pairs."""
    for p in ['A','B']:
        print(f"metrics for {p} images")
        for k in metrics1[p]:
            s = f"\t{k[0]} - {k[1]}: {metrics1[p][k]:.4g}"
            if metrics2 is not None:
                s += f", {metrics2[p][k]:.4g}"
            print(s)
        print()


def plot_metrics(metrics, labels, title):
    """Plot metrics."""
    metrics_std = None
    if isinstance(metrics, list):
        metrics_mean, metrics_std = metrics
        metrics = metrics_mean
    for p in ['A','B']:
        table = metric_dict_to_table(metrics[p], labels)
        points_2d = create_2d_map(table)
        plot_2d_map(
            points_2d,
            labels,
            file_path=BASE_FOLDER / f'docs/assets/evaluation/{title.lower()}_map_images_{p}.png',
            title=f'{title.upper()} for {p} Images')
        plot_2d_map(
            points_2d,
            labels,
            file_path=BASE_FOLDER / f'docs/assets/evaluation/{title.lower()}_map_images_{p}_legend.png',
            title=f'{title.upper()} for {p} Images',
            label_dist=0)

        data = {
            'class': labels[1:],
            'value': [metrics[p][(labels[0],k)] for k in labels[1:]]
        }
        if metrics_std is not None:
            data['std'] = [metrics_std[p][(labels[0],k)] for k in labels[1:]]
        plot_hbar(
            data,
            file_path=BASE_FOLDER / f'docs/assets/evaluation/{title.lower()}_bar_images_{p}.png',
            title=f'{title.upper()} for {p} Images',
            x_label=f'{title.upper()} to Real Images',
            y_label='')


def plot_histograms(metrics, labels, title):
    """Plot histograms."""
    for p in ['A','B']:
        metrics_list = []
        for k in labels[1:]:
            metrics_list.append(metrics[p][('Real',k)].flatten())
        plot_stacked_histograms(
            metrics_list,
            labels[1:],
            file_path=BASE_FOLDER / f'docs/assets/evaluation/{title.lower()}_histograms_{p}.png',
            title=f'{title.upper()} for {p} Images', xlabel=f'{title.upper()} to Real Images',
            figsize=(8, 1.5))


def save_samples(real_image_list, real_class, models):
    """Save image tranlation samples to file."""
    def img_from_name(img_name, model):
        if model == 'Real':
            img_path = BASE_FOLDER / f'data/external/nexet/input_{real_class}' / img_name
        else:
            img_path = BASE_FOLDER / f'data/external/nexet/output_{real_class}_{model}' / img_name
        image = Image.open(img_path).convert('RGB')
        image = transforms.ToTensor()(image)
        return image

    images = []
    for img_name in real_image_list:
        images.append(img_from_name(img_name, 'Real'))
    for v in models.values():
        for img_name in real_image_list:
            images.append(img_from_name(img_name, v))

    images_tensor = torch.stack(images)

    ImageTools.show_img(
        images_tensor,
        title=f'Translation Samples for {real_class} Images',
        figsize = (20, 3*len(models)), nrow=len(real_image_list),
        labels=['Real'] + list(models),
        rotation=0
    )
    plt.savefig(BASE_FOLDER / f'docs/assets/evaluation/Samples_{real_class}.png')
    plt.close()


def save_metrics(metrics, file_name):
    """Save metrics to file."""
    file_path = BASE_FOLDER / f'docs/assets/evaluation/{file_name}'
    with open(file_path, 'wb') as file:
        pickle.dump(metrics, file)

def load_metrics(file_name):
    """Load metrics from file."""
    file_path = BASE_FOLDER / f'docs/assets/evaluation/{file_name}'
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def main():
    """Main function."""

    n_tests = 7
    test_cases_to_build_images = [] # Indexes of test cases to build images
    recalculate_metrics = False # If False, will load metrics from pkl files
    n_samples = 5
    # best_model = 5 # Index of the 'best' model


    # Build translated images
    for i in test_cases_to_build_images:
        build_images(i)


    # Build image data loaders
    model_list = {
        'Real': 'real',
        # 'Oposite class': 'oposite',
        'CycleGAN': 'cyclegan',
        'CycleGAN-turbo': 'turbo',
    }
    for i in range(1, n_tests+1):
        test_case = TEST_CASES[str(i)]
        model_list[test_case['short_description']] = f'test_{i}'

    data_loaders = {}
    for k,v in model_list.items():
        data_loaders[k] = build_data_loaders(v)
    labels = list(model_list.keys())


    print('========= FID =========')
    if recalculate_metrics:
        fid_metrics = get_fid(data_loaders)
        save_metrics(fid_metrics, 'fid_metrics.pkl')
    else:
        fid_metrics = load_metrics('fid_metrics.pkl')
    print_metric_pairs(fid_metrics)
    plot_metrics(fid_metrics, labels, 'FID')


    print('========= LPIPS =========')
    if recalculate_metrics:
        lpips_metrics = get_lpips(data_loaders)
        save_metrics(lpips_metrics, 'lpips_metrics.pkl')
    else:
        lpips_metrics = load_metrics('lpips_metrics.pkl')
    plot_histograms(lpips_metrics, labels, 'LPIPS')
    lpips_metrics_mean = transform_metrics(lpips_metrics, transform=lambda x: float(x.mean()))
    lpips_metrics_std = transform_metrics(lpips_metrics, transform=lambda x: float(x.std()))
    print_metric_pairs(lpips_metrics_mean, lpips_metrics_std)
    plot_metrics([lpips_metrics_mean, lpips_metrics_std], labels, 'LPIPS')

    lpips_metrics_dist = lpips_distance(lpips_metrics_mean, lpips_metrics_std)
    print("LPIPS 'distances'")
    print_metric_pairs(lpips_metrics_dist)
    # labels.remove('Oposite class')
    plot_metrics(lpips_metrics_dist, labels, 'W-LPIPS')

    # Save samples
    for p in ['A','B']:
        images_csv = BASE_FOLDER / f'data/external/nexet/input_{p}_all_filtered.csv'
        df = pd.read_csv(images_csv)
        img_list = df['file_name'].sample(n_samples).tolist()
        models = model_list.copy()
        models.pop('Real', None)
        # models.pop('Oposite class', None)
        save_samples(img_list, p, models)

    # Calculate LPIPS
    #   Sample images along histogram
    #       Define the 'best' test case
    #       Get the mean distance from each model image to the real images
    #       Order images and sample evenly along the histogram
    #       Plot histogram of distances with samples images

if __name__ == '__main__':
    main()
