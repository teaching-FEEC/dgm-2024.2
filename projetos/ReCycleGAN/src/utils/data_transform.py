"""Module with class to build image dataset with transformations."""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .utils import remove_all_files

class Images():
    """Class that holds functions to build image dataset."""
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
    def _append_to_filename(file_path, suffix):
        file = Path(file_path)
        new_name = file.stem + suffix + file.suffix
        new_file = file.with_name(new_name)
        return new_file

    @staticmethod
    def _split_dataset(img_list, csv_file, split=0.7, seed=42):
        img_train, img_test = train_test_split(
            img_list, train_size=split, random_state=seed, shuffle=True)

        df = pd.DataFrame({'file_name': img_train})
        csv_file_train = Images._append_to_filename(csv_file, '_train')
        df.to_csv(csv_file_train, index=False)

        df = pd.DataFrame({'file_name': img_test})
        csv_file_test = Images._append_to_filename(csv_file, '_test')
        df.to_csv(csv_file_test, index=False)

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
        img_ok = Images._transform_images(
            img_list=img_list,
            img_folder=img_folder,
            output_folder=output_folder,
            transformation_function=transformation_function,
            transformation_params=transformation_params)
        Images._split_dataset(
            img_list=img_ok,
            csv_file=Path(output_folder).parent / f'{Path(output_folder).name}.csv',
            split=split,
            seed=random_seed)
