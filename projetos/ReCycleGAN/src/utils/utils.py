"""Assorted functions."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


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
