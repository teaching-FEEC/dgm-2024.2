# pylint: disable=invalid-name
"""Assorted functions."""

import subprocess
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import wandb
import pynvml

class Constants:
    """Project constants"""
    DATASET_FILEPATH = "./data/external"
    WB_PROJECT = "cyclegan"
    WB_DB_UPLOAD_JOB = "dataset_upload"
    WB_DB_ARTIFACT_TYPE = "datasets"

def remove_all_files(folder_path):
    """Remove all files in a folder."""
    folder = Path(folder_path)
    if folder.exists() and folder.is_dir():
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                remove_all_files(file)

def filter_dataframe(df, filter_dict):
    """Filter a DataFrame by multiple columns.

    Attributes:
    ------------
    df: pd.DataFrame
        DataFrame to filter.
    filter_dict: dict
        Dictionary with column names as keys and values to filter as values.
    """
    mask = pd.Series([True] * len(df))
    for col, values in filter_dict.items():
        mask &= df[col].isin(values)
    return df[mask]

def img_size_count(img_folder, extension='jpg', verbose=False):
    """Count the number of images by size in a folder.

    Attributes:
    ------------
    img_folder: str
        Path to the folder containing images.
    extension: str
        Image file extension.
        (Default: 'jpg')
    """
    sizes = []
    img_folder = Path(img_folder)
    for img_path in tqdm(list(img_folder.glob(f'*.{extension}'))):
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
        except Image.UnidentifiedImageError:
            if verbose:
                print(f'Error opening image: {img_path}')
    df_sizes = pd.DataFrame(sizes, columns=['width', 'height'])
    return df_sizes.value_counts().reset_index(name='count')


def img_size_count_plot(size_counts, figsize=(4, 3), show=False):
    """Plot the number of images by size."""
    fig, axs = plt.subplots(1,1,figsize=figsize)
    tick_label = [f"{w}x{h}" for w, h in size_counts[['width', 'height']].values]
    axs.bar(size_counts.index, size_counts['count'], tick_label=tick_label, alpha=0.7, color='blue')
    axs.set_xlabel('Image Size (width x height)')
    axs.set_ylabel('Count')
    axs.set_title('Number of Images by Size')
    plt.tight_layout()
    if show:
        plt.show()
        return None
    return fig


def resize_and_crop(image_path, output_path, target_size, size_filter=None):
    """
    Resize and crop an image to fit the target size.

    Image is cropped at the center to match the target size aspect ratio.

    Parameters:
    ------------
    image_path: str
        Path to the input image file.
    output_path: str
        Path to save the output image file.
    target_size: tuple
        (width, height) of the target size.
    size_filter: list of tuples, optional
        [(width, height)] of input image sizes to resize.
        If None, resize all images.
        (default=None)

    Returns:
    ---------
    bool
        True if the image was resized, False otherwise.
    """
    with Image.open(image_path) as img:

        original_width, original_height = img.size
        target_width, target_height = target_size

        if size_filter is not None:
            if (original_width, original_height) not in size_filter:
                return False

        gray_img = img.convert('L')
        np_gray = np.array(gray_img)
        mask = np_gray > 10
        coords = np.argwhere(mask)
        if coords.size > 0:
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1

            if (x1-x0<target_width) or (y1-y0<target_height):
                return False

            bbox = (y0, x0, y1, x1)
            img = img.crop(bbox)
            original_width, original_height = img.size
        else:
            return False


        original_aspect = original_width / original_height
        target_aspect = target_width / target_height

        if original_aspect > target_aspect:
            new_height = target_height
            new_width = int(new_height * original_aspect)
        else:
            new_width = target_width
            new_height = int(new_width / original_aspect)

        left = (new_width - target_width) / 2
        top = (new_height - target_height) / 2
        right = left + target_width
        bottom = target_height

        img = img.resize((new_width, new_height), Image.LANCZOS) # pylint: disable=no-member
        img = img.crop((left, top, right, bottom))
        img.save(output_path)

        return True
    return False

def show_img(img, title=None, figsize=(4, 3), show=False, change_scale=False, nrow=None, labels=None):
    """Show an image using matplotlib.

    Attributes:
    ------------
    img: torch.Tensor or np.ndarray
        Image tensor or array.
    title: str
        Title of the image.
    figsize: tuple
        Figure size (width, height).
        (Default: (4, 3))
    show: bool
        Whether to display the image.
        (Default: False)
    change_scale: bool
        Whether to change the scale of the image
        from [-1, 1] to [0, 1].
    nrow: int
        Number of images per row to display if the image is a tensor.
        If None, the number of rows is calculated based on the
        number of images in the tensor.
        (Default: None)
    labels: list of str
        List of labels to display along the vertical axis.
        If None, no labels are displayed.
        (Default: None)
    """

    width, height = figsize
    f_size = int(min(width, height) * 1.5)

    if change_scale:
        img = (img + 1) / 2
    if len(img.shape) > 4:
        msg = 'Image tensor has more than 4 dimensions.'
        raise ValueError(msg)
    if len(img.shape) == 4:
        if nrow is None:
            nrow = int(max(4, min(8, np.ceil(img.shape[0] / 2))))
        grid = make_grid(img, nrow=nrow, normalize=False, scale_each=False)
        return show_img(grid, title=title, figsize=figsize, show=show, labels=labels)

    img = img.permute(1, 2, 0)
    if img.shape[2]==1:
        img = img.view(img.shape[0], img.shape[1])

    fig, axs = plt.subplots(1,1,figsize=figsize)
    if title is not None:
        axs.set_title(title, fontsize=f_size+2, fontweight='bold')
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    axs.imshow(img)

    if labels is None:
        axs.axis('off')
    else:
        axs.xaxis.set_visible(False)
        num_labels = len(labels) * 2 + 1
        y_ticks = np.linspace(0, img.shape[0] - 1, num_labels)
        y_lab = [labels[i//2] if i % 2 == 1 else '' for i in range(num_labels)]
        axs.set_yticks(ticks=y_ticks, labels=y_lab, rotation=90, fontsize=f_size)

    plt.tight_layout()
    if show:
        plt.show()
        return None
    return fig

def image_folder_to_tensor(img_dir, img_size=(256, 256), img_glob='*'):
    """
    Reads all images in a folder and puts them into a single PyTorch tensor.

    Parameters:
    ------------
    img_dir: str or Path
        Path to the directory containing images.
    img_size: tuple, optional
        Desired size of the images (width, height).
        (default=(256, 256))
    img_glob: str, optional
        Glob pattern to filter images.
        (default='*')

    Returns:
    ------------
        torch.Tensor: A tensor containing all images.
    """
    img_dir = Path(img_dir)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    image_tensors = []
    for img_path in img_dir.glob(img_glob):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            image_tensors.append(img_tensor)

    all_images_tensor = torch.stack(image_tensors)
    return all_images_tensor

def save_model(model, local_path='model.pth', wandb_log=True):
    """
    Saves the model state to a local file and optionally logs it to Weights & Biases (WandB).

    Saves a checkpoint of the model's state dictionary to the specified local file.
    If `wandb_log` is True, the model will also be saved to WandB for remote logging.

    Parameters:
    ------------
    model: torch.nn.Module
        Model instance to save.
    local_path: str
        File path where the model will be saved locally.
        (default: 'model.pth')
    wandb_log: bool
        Whether to log the model to WandB for version control
        and experiment tracking.
        (default: True)
    """
    # Save locally
    torch.save(model.state_dict(), local_path)

    # Save to WandB
    if wandb_log:
        wandb.save(local_path)

def save_losses(loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id, loss_G_plp, filename='losses.txt'):
    """
    Saves the generator and discriminator losses to a text file.

    Saves a text file containing the losses for the generator and discriminators
    (A and B) over the training epochs.

    Args:
    - loss_G (list): List of generator losses over the training epochs.
    - loss_D_A (list): List of discriminator A losses over the training epochs.
    - loss_D_B (list): List of discriminator B losses over the training epochs.
    - filename (str): The file path where the losses will be saved. Defaults to 'losses.txt'.
    """
    np.savetxt(
        filename,
        np.column_stack((loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id, loss_G_plp)),
        header='Generator total loss, Discriminator A loss, Discriminator B loss')

def train_one_epoch(epoch, model, train_A, train_B, device, n_samples=None, plp_step=0):
    """
    Trains the CycleGAN model for a single epoch and returns the generator and discriminator losses.

    Args:
    - epoch (int): The current epoch number.
    - model (CycleGAN): The CycleGAN model instance.
    - train_A (DataLoader): DataLoader for domain A training images.
    - train_B (DataLoader): DataLoader for domain B training images.
    - device (torch.device): The device on which the model and data are
    loaded (e.g., 'cuda' or 'cpu').
    - n_samples (int): Number of samples to train on per batch.
    If None, train on all samples. Default is None.
    - plp_step: Steps between Path Length Penalty calculations. Used to adjust
    PLP loss value. Default is 0.

    Returns:
    - loss_G (float): The total loss of the generator for this epoch.
    - loss_D_A (float): The total loss of discriminator A for this epoch.
    - loss_D_B (float): The total loss of discriminator B for this epoch.

    During training:
    - It iterates through the batches of both domains (A and B) and performs
    optimization on the generators and discriminators.
    - Progress is tracked with a `tqdm` progress bar that shows current generator
    and discriminator losses.
    """

    progress_bar = tqdm(zip(train_A, train_B), desc=f'Epoch {epoch:03d}',
                        leave=False, disable=False)

    loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id, loss_G_plp = 0, 0, 0, 0, 0, 0, 0
    for batch_A, batch_B in progress_bar:
        progress_bar.set_description(f'Epoch {epoch:03d}')

        if n_samples is not None:
            batch_A = batch_A[:n_samples]
            batch_B = batch_B[:n_samples]

        real_A = batch_A.to(device)
        real_B = batch_B.to(device)

        # Perform one optimization step
        loss = model.optimize(real_A, real_B)
        loss_G += loss.loss_G.item()
        loss_D_A += loss.loss_D_A.item()
        loss_D_B += loss.loss_D_B.item()
        loss_G_ad += loss.loss_G_ad.item()
        loss_G_cycle += loss.loss_G_cycle.item()
        loss_G_id += loss.loss_G_id.item()
        loss_G_plp += loss.loss_G_plp.item()

        progress_bar.set_postfix({
            'G_loss': f'{loss.loss_G.item():.4f}',
            'D_A_loss': f'{loss.loss_D_A.item():.4f}',
            'D_B_loss': f'{loss.loss_D_B.item():.4f}'
        })
    progress_bar.close()

    loss_G /= (len(train_A) + len(train_B)) / 2
    loss_D_A /= len(train_A)
    loss_D_B /= len(train_B)
    loss_G_ad /= (len(train_A) + len(train_B)) / 2
    loss_G_cycle /= (len(train_A) + len(train_B)) / 2
    loss_G_id /= (len(train_A) + len(train_B)) / 2
    loss_G_plp /= (len(train_A) + len(train_B)) / 2 * plp_step

    msg = f'Epoch {epoch:03d}: G_loss={loss_G:.4g}, '
    msg += f'D_A_loss={loss_D_A:.4g}, D_B_loss={loss_D_B:.4g}, '
    msg += f'G_ad={loss_G_ad:.4g}, G_cycle={loss_G_cycle:.4g}, '
    msg += f'G_id={loss_G_id:.4g}, G_plp={loss_G_plp:.4g}'
    print(msg)
    return loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id, loss_G_plp

# Plot losses
def plot_losses(train_losses, val_losses):
    """
    Plots the training and validation losses over the epochs.

    Args:
    - train_losses (list): List of training losses (e.g., generator losses) over the epochs.
    - val_losses (list): List of validation losses over the epochs.

    Displays:
    - A line plot showing the progression of training and validation losses.
    - Training and validation losses are plotted against the number of epochs.
    """
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label='Training Loss',
        linewidth=2, alpha=0.7)
    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label='Validation Loss',
        linewidth=2, alpha=0.7)
    plt.title('CycleGAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def get_gpu_memory_usage():
    """Get list of dict with memory usage of all GPUs."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    gpu_memory_info = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        gpu_memory_info.append({
            'gpu_index': i,
            'total_memory': memory_info.total,
            'used_memory': memory_info.used,
            'free_memory': memory_info.free
        })
    pynvml.nvmlShutdown()
    return gpu_memory_info

def print_gpu_memory_usage(msg=None, short_msg=False):
    """Print the memory usage of all GPUs.

    Attibutes:
    ------------
    msg: str, optional
        Message to print before the memory usage. If None
        provided, the default message is "GPU Memory Usage".
        (default=None)
    short_msg: bool
        If True, prints a single line message with the total
        memory usage across all GPUs.
        (default=False)
    """
    gpu_memory_info = get_gpu_memory_usage()
    if short_msg:
        if msg is None:
            msg = "GPU Memory Usage"
        total = sum(info['total_memory'] for info in gpu_memory_info)
        used = sum(info['used_memory'] for info in gpu_memory_info)
        print(f"{msg}: {used / (1024 ** 2):.2f} MB ({used / total * 100:.2f}% used)")
        return

    ident = ""
    if msg is not None:
        print(msg)
        ident = " " * 2
    if len(gpu_memory_info) == 0:
        print(f"{ident}No GPUs found.")
    for info in gpu_memory_info:
        print(f"{ident}GPU {info['gpu_index']}:")
        print(f"{ident}  Total Memory: {info['total_memory'] / (1024 ** 2):.2f} MB")
        print(f"{ident}  Used Memory: {info['used_memory'] / (1024 ** 2):.2f} MB")
        print(f"{ident}  Free Memory: {info['free_memory'] / (1024 ** 2):.2f} MB")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Attributes:
    ------------
    model: nn.Module
        The PyTorch model.

    Returns:
    ------------
        The total number of parameters: int
    """
    return sum(p.numel() for p in model.parameters())

def get_current_commit():
    """Get current git hash of the repository."""
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        commit_message = subprocess.check_output(['git', 'log', '-1', '--pretty=%B']).strip().decode('utf-8')
        return git_hash, commit_message
    except subprocess.CalledProcessError:
        return None, None

def save_dict_as_json(data, file_path):
    """
    Save a dictionary as a formatted JSON file.

    Parameters:
    ------------
    data: dict
        The dictionary to save.
    file_path: str
        The path to the output JSON file.
    """
    out = {}
    for k,v in data.items():
        out[k] = str(v)

    # out_file = open(file_path, "w")
    # json.dump(out, out_file, indent = 6)
    # out_file.close()

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(out, json_file, indent=4, sort_keys=True)
