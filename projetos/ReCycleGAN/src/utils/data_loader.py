"""Module that holds the image DataLoader."""

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms
from PIL import Image
import pandas as pd

class ImageDataset(Dataset):
    """Custom image Dataset."""
    def __init__(self, csv_file, img_dir, transformation=None, file_name_col='file_name'):
        if isinstance(csv_file, list):
            self.image_paths = csv_file
        else:
            self.image_paths = pd.read_csv(csv_file)[file_name_col].tolist()
        self.img_dir = Path(img_dir)
        self.transformation = transformation

    def __len__(self):
        return len(self.image_paths)

    def set_len(self, new_len):
        """Set the length of the dataset."""
        self.image_paths = self.image_paths[:new_len]

    def __getitem__(self, idx):
        img_path = self.img_dir / self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transformation:
            image = self.transformation(image)
        else:
            image = transforms.ToTensor()(image)

        return image

def get_img_dataloader(csv_file, img_dir=None, transformation=None, file_name_col='file_name',
                       batch_size=32, shuffle=True, num_workers=1):
    """Get image DataLoader.

    Parameters:
    ------------
    csv_file: str
        Path to the CSV file containing image names.
    img_dir: str, optional
        Path to the images folder. If None, uses the folder
        with the same name as the CSV file, deleting
        '_train' and '_test'.
        (Default: None)
    transformation: torchvision.transforms
        Image transformation.
        (Default: None)
    file_name_col: str
        Column name in the CSV file containing the image names.
        (Default: 'file_name')
    batch_size: int
        Batch size.
        (Default: 32)
    shuffle: bool
        Shuffle the data.
        (Default: True)
    num_workers: int
        Number of workers.
        (Default: 1)
    """
    if img_dir is None:
        stem = Path(csv_file).stem
        if '_test' in stem:
            stem = stem.split('_test')[0]
        elif '_train' in stem:
            stem = stem.split('_train')[0]

        img_dir = Path(csv_file).parent / stem
    if not Path(img_dir).exists():
        msg = f"Folder {img_dir} not found."
        raise FileNotFoundError(msg)
    dataset = ImageDataset(csv_file, img_dir, transformation, file_name_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def copy_dataloader(data_loader):
    """Copy a DataLoader."""
    dataset = ImageDataset(
        csv_file=data_loader.dataset.image_paths,
        img_dir=data_loader.dataset.img_dir,
        transformation=data_loader.dataset.transformation
    )
    new_loader = DataLoader(
        dataset,
        batch_size=data_loader.batch_size,
        shuffle=is_dataloader_shuffled(data_loader),
        num_workers=data_loader.num_workers
    )
    return new_loader

def is_dataloader_shuffled(dataloader):
    """
    Check if a DataLoader has its elements shuffled.

    Parameters:
    ------------
    dataloader: DataLoader
        The DataLoader to check.

    Returns:
    ------------
    bool
        True if the DataLoader is shuffled, False otherwise.
    """
    if isinstance(dataloader.sampler, RandomSampler):
        return True
    elif isinstance(dataloader.sampler, SequentialSampler):
        return False
    else:
        # Custom sampler, check if it has a shuffle attribute
        return getattr(dataloader.sampler, 'shuffle', False)
