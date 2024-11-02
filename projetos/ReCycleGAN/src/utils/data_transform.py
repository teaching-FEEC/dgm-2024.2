# pylint: disable=import-error
"""Module with class to build image dataset with transformations."""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split

from .utils import remove_all_files


class ImageTools():
    """Class that holds image handling tools."""
    def __init__(self):
        pass

    @staticmethod
    def _transform_images(img_list, img_folder, output_folder,
                          transformation_function, transformation_params):
        f = Path(output_folder)
        if not f.exists():
            f.mkdir(parents=True)
        else:
            remove_all_files(f)

        src_folder = Path(img_folder)
        img_ok = []
        for img in tqdm(img_list):
            src = src_folder / img
            dest = f / img
            if transformation_function(src, dest, **transformation_params):
                img_ok.append(img)
        return img_ok


    @staticmethod
    def _split_dataset(img_list, csv_file, split=0.7, seed=42):
        img_train, img_test = train_test_split(
            img_list, train_size=split, random_state=seed, shuffle=True)

        csv_file = Path(csv_file)

        df_train = pd.DataFrame({'file_name': img_train})
        new_name = csv_file.stem + '_train' + csv_file.suffix
        csv_file_train = csv_file.with_name(new_name)
        df_train.to_csv(csv_file_train, index=False)

        df_test = pd.DataFrame({'file_name': img_test})
        new_name = csv_file.stem + '_test' + csv_file.suffix
        csv_file_test = csv_file.with_name(new_name)
        df_test.to_csv(csv_file_test, index=False)

        df_all = pd.concat([df_train, df_test], axis=0)
        new_name = csv_file.stem + '_all' + csv_file.suffix
        csv_file_test = csv_file.with_name(new_name)
        df_all.to_csv(csv_file_test, index=False)


    @staticmethod
    def build_dataset(img_list, img_folder, output_folder, transformation_function,
                      transformation_params, split, random_seed):
        """Build image dataset with transformations.

        Parameters:
        ------------
        img_list: [str]
            List of image filenames.
        img_folder: str
            Path to the folder containing the images.
        output_folder: str
            Path to the folder to save the transformed images.
        transformation_function: function
            Function to transform the images. Must be in the form:
            transformation_function(src, dest, **transformation_params)
            Must return True if successful, False otherwise.
        transformation_params: dict
            Parameters to pass to the transformation function.
        split: float
            Percentage of images to use for training.
        random_seed: int
            Random seed for reproducibility.
        """
        img_ok = ImageTools._transform_images(
            img_list=img_list,
            img_folder=img_folder,
            output_folder=output_folder,
            transformation_function=transformation_function,
            transformation_params=transformation_params)
        ImageTools._split_dataset(
            img_list=img_ok,
            csv_file=Path(output_folder).parent / f'{Path(output_folder).name}.csv',
            split=split,
            seed=random_seed)
        return img_ok


    @staticmethod
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


    @staticmethod
    def img_size_count_plot(size_counts, figsize=(4, 3)):
        """Plot the number of images by size."""
        fig, axs = plt.subplots(1,1,figsize=figsize)
        tick_label = [f"{w}x{h}" for w, h in size_counts[['width', 'height']].values]
        axs.bar(size_counts.index,
                size_counts['count'],
                tick_label=tick_label,
                alpha=0.7, color='blue')
        axs.set_xlabel('Image Size (width x height)')
        axs.set_ylabel('Count')
        axs.set_title('Number of Images by Size')
        plt.tight_layout()
        return fig


    @staticmethod
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

            def _remove_black_borders(img, target_width, target_height):
                gray_img = img.convert('L')
                np_gray = np.array(gray_img)
                mask = np_gray > 10
                coords = np.argwhere(mask)
                if coords.size > 0:
                    x0, y0 = coords.min(axis=0)
                    x1, y1 = coords.max(axis=0) + 1

                    if (x1-x0<target_width) or (y1-y0<target_height):
                        return None

                    bbox = (y0, x0, y1, x1)
                    return img.crop(bbox)
                return None

            def _resize_and_crop(img, target_width, target_height):
                original_width, original_height = img.size
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
                return img

            original_width, original_height = img.size
            if size_filter is not None:
                if (original_width, original_height) not in size_filter:
                    return False

            target_width, target_height = target_size
            img = _remove_black_borders(img, target_width, target_height)
            if img is None:
                return False

            img = _resize_and_crop(img, target_width, target_height)
            img.save(output_path)
            return True

        return False


    @staticmethod
    def show_img(img, title=None, figsize=(4, 3), nrow=None, labels=None):
        """Show a group of images using matplotlib.

        If negative values are present in the image tensor, it is assumed
        the scale is [-1, 1], and is then changed to [0, 1].

        Attributes:
        ------------
        img: torch.Tensor or np.ndarray
            Image tensor or array.
        title: str
            Title of the image.
        figsize: tuple
            Figure size (width, height).
            (Default: (4, 3))
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
        if len(img.shape) > 4:
            msg = 'Image tensor has more than 4 dimensions.'
            raise ValueError(msg)
        if img.min() < 0:
            img = (img + 1) / 2

        width, height = figsize
        f_size = int(min(width, height) * 1.5)

        if len(img.shape) == 4:
            if nrow is None:
                nrow = int(max(4, min(8, np.ceil(img.shape[0] / 2))))
            grid = make_grid(img, nrow=nrow, normalize=False, scale_each=False)
            return ImageTools.show_img(grid, title=title, figsize=figsize, labels=labels)

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
        return fig


    @staticmethod
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
