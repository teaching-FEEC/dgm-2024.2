"""Assorted functions."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import torch
import wandb
import pynvml

class Constants:
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
        if size_filter is not None:
            if (original_width, original_height) not in size_filter:
                return False

        target_width, target_height = target_size

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

def show_img(img, title=None, figsize=(4, 3), show=False, change_scale=False):
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
    """
    if change_scale:
        img = (img + 1) / 2
    if len(img.shape) > 4:
        msg = 'Image tensor has more than 4 dimensions.'
        raise ValueError(msg)
    if len(img.shape) == 4:
        nrow = max(4, min(8, np.ceil(img.shape[0] / 2)))
        grid = make_grid(img, nrow=nrow, normalize=False, scale_each=False).cpu()
        return show_img(grid, title=title, figsize=figsize, show=show)

    img = img.permute(1, 2, 0)
    if img.shape[2]==1:
        img = img.view(img.shape[0], img.shape[1])

    fig, axs = plt.subplots(1,1,figsize=figsize)
    if title is not None:
        axs.set_title(title)
    axs.imshow(img)
    axs.axis('off')
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

    Args:
    - model (torch.nn.Module): The model instance to save.
    - local_path (str): The file path where the model will be saved locally. Defaults to 'model.pth'.
    - wandb_log (bool): Whether to log the model to WandB for version control and experiment tracking. Defaults to True.

    Saves:
    - A checkpoint of the model's state dictionary to the specified local file.
    - If `wandb_log` is True, the model will also be saved to WandB for remote logging.
    """
    # Save locally
    torch.save(model.state_dict(), local_path)

    # Save to WandB
    if wandb_log:
        wandb.save(local_path)

def save_losses(loss_G, loss_D_A, loss_D_B, filename='losses.txt'):
    """
    Saves the generator and discriminator losses to a text file.

    Args:
    - loss_G (list): List of generator losses over the training epochs.
    - loss_D_A (list): List of discriminator A losses over the training epochs.
    - loss_D_B (list): List of discriminator B losses over the training epochs.
    - filename (str): The file path where the losses will be saved. Defaults to 'losses.txt'.

    Saves:
    - A text file containing the losses for the generator and discriminators (A and B) over the training epochs.
    """
    np.savetxt(filename, np.column_stack((loss_G, loss_D_A, loss_D_B)), header='Generator total loss, Discriminator A loss, Discriminator B loss')

def train_one_epoch(epoch, model, train_A, train_B, device):
    """
    Trains the CycleGAN model for a single epoch and returns the generator and discriminator losses.

    Args:
    - epoch (int): The current epoch number.
    - model (CycleGAN): The CycleGAN model instance.
    - train_A (DataLoader): DataLoader for domain A training images.
    - train_B (DataLoader): DataLoader for domain B training images.
    - device (torch.device): The device on which the model and data are loaded (e.g., 'cuda' or 'cpu').

    Returns:
    - loss_G (float): The total loss of the generator for this epoch.
    - loss_D_A (float): The total loss of discriminator A for this epoch.
    - loss_D_B (float): The total loss of discriminator B for this epoch.

    During training:
    - It iterates through the batches of both domains (A and B) and performs optimization on the generators and discriminators.
    - Progress is tracked with a `tqdm` progress bar that shows current generator and discriminator losses.
    """

    progress_bar = tqdm(zip(train_A, train_B), desc=f'Epoch {epoch:03d}', leave=False)

    for batch_A, batch_B in progress_bar:
        real_A = batch_A[0].to(device)
        real_B = batch_B[0].to(device)

        # Perform one optimization step
        loss_G, loss_D_A, loss_D_B = model.optimize(real_A, real_B)

        progress_bar.set_postfix({
            'G_loss': f'{loss_G:.4f}',
            'D_A_loss': f'{loss_D_A:.4f}',
            'D_B_loss': f'{loss_D_B:.4f}'
        })

    return loss_G, loss_D_A, loss_D_B

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
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2, alpha=0.7)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linewidth=2, alpha=0.7)
    plt.title('CycleGAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def get_gpu_memory_usage():
    """Get the memory usage of all GPUs."""
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
    """Print the memory usage of all GPUs."""
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
