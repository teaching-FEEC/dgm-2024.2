from torch.utils.data import Dataset
import torch
import numpy as np
from glob import glob
import SimpleITK as sitk
from lungmask import LMInferer
from typing import Optional, Callable
import os


class rawCTData(Dataset):
    def __init__(self,
                 raw_data_folder: str,
                 mode: str = 'train',
                 transform: Optional[Callable] = None):
        super().__init__()
        if mode == 'train':
            self.cts = sorted(glob(os.path.join(raw_data_folder,
                                                mode,
                                                "imagesTr",
                                                "*.nii.gz")))
            self.labels = sorted(glob(os.path.join(raw_data_folder,
                                                   mode,
                                                   "labelsTr",
                                                   "*.nii.gz")))
        else:
            self.cts = sorted(glob(os.path.join(raw_data_folder,
                                                mode,
                                                "*.nii.gz")))
        self.transform = transform
        self.inferer = LMInferer()

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, idx: int):
        '''
        Carregar, transformar e retornar o item 'i' do dataset
        '''
        ct_path = self.cts[idx]
        ct_labels_path = self.labels[idx]

        # Ler imagem usando a biblioteca SITK
        print(f'Reading {ct_path} and {ct_labels_path}.......')
        image = sitk.ReadImage(ct_path)
        label = sitk.ReadImage(ct_labels_path)

        # Converter imagem para array numpy na variável 'ct'
        print("Converting to array")
        ct = sitk.GetArrayFromImage(image)
        ct_label = sitk.GetArrayFromImage(label)
        ct_lung = self.inferer.apply(ct)
        ct_lung[ct_lung > 1] = 1

        # Se uma função de transformada foi passada para o dataset, aplicá-la
        if self.transform is not None:
            ct = self.transform(ct)
        print(ct.shape)
        # Retornar a imagem e metadados
        return ct, ct_label, ct_lung


class lungCTData(Dataset):
    def __init__(self, processed_data_folder: str,
                 mode: Optional[str] = None,
                 start: Optional[int] = None,
                 end: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 **kwargs):
        super().__init__()
        if start is not None and end is not None:
            if mode is not None:
                self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "images",
                                                    "*.npz")))[start:end]
                self.labels = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "lungs",
                                                    "*.npz")))[start:end]
            else:
                self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                    "images",
                                                    "*.npz")))[start:end]
                self.labels = sorted(glob(os.path.join(processed_data_folder,
                                                    "lungs",
                                                    "*.npz")))[start:end]
        elif start is not None and end is None:
            if mode is not None:
                self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "images",
                                                    "*.npz")))[start:]
                self.labels = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "lungs",
                                                    "*.npz")))[start:]
            else:
                self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                    "images",
                                                    "*.npz")))[start:]
                self.labels = sorted(glob(os.path.join(processed_data_folder,
                                                    "lungs",
                                                    "*.npz")))[start:]
        elif start is None and end is not None:
            if mode is not None:
                self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "images",
                                                    "*.npz")))[:end]
                self.labels = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "lungs",
                                                    "*.npz")))[:end]
            else:
                self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                    "images",
                                                    "*.npz")))[:end]
                self.labels = sorted(glob(os.path.join(processed_data_folder,
                                                    "lungs",
                                                    "*.npz")))[:end]
        else:
            if mode is not None:
                self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "images",
                                                    "*.npz")))
                self.labels = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "lungs",
                                                    "*.npz")))
            else:
                self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                    "images",
                                                    "*.npz")))
                self.labels = sorted(glob(os.path.join(processed_data_folder,
                                                    "lungs",
                                                    "*.npz")))
        self.transform = transform(**kwargs) if transform is not None else None
        assert len(self.cts) == len(self.labels)

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, idx: int):
        '''
        Carregar, transformar e retornar o item 'i' do dataset
        '''
        ct_path = self.cts[idx]
        ct_labels_path = self.labels[idx]

        # Ler imagem usando a biblioteca SimpleITK
        # O objeto image contêm tambem metadados
        # print(f'Reading {ct_path} and {ct_labels_path}.......')
        image_npz = np.load(ct_path)
        lung_npz = np.load(ct_labels_path)

        ct = image_npz['arr_0']
        lung = lung_npz['arr_0']
        ct = torch.tensor(ct).to(torch.float32)
        ct = ct.unsqueeze(0)
        lung = torch.tensor(lung).to(torch.float32).unsqueeze(0)

        # Se uma função de transformada foi passada para o dataset, aplicá-la
        if self.transform is not None:
            lung = self.transform(lung)
        # Retornar a imagem e metadados
        return ct, lung


class processedCTData(Dataset):
    def __init__(self, processed_data_folder: str,
                 mode: str = 'train',
                 start: Optional[int] = None,
                 end: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 **kwargs):
        super().__init__()
        if start is not None and end is not None:
            self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                mode,
                                                "images",
                                                "*.npz")))[start:end]
            self.airways = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "labels",
                                                    "*.npz")))[start:end]
            self.lungs = sorted(glob(os.path.join(processed_data_folder,
                                                   mode,
                                                   "lungs",
                                                   "*.npz")))[start:end]
        elif start is not None and end is None:
            self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                mode,
                                                "images",
                                                "*.npz")))[start:]
            self.airways = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "labels",
                                                    "*.npz")))[start:]
            self.lungs = sorted(glob(os.path.join(processed_data_folder,
                                                   mode,
                                                   "lungs",
                                                   "*.npz")))[start:]
        elif start is None and end is not None:
            self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                mode,
                                                "images",
                                                "*.npz")))[:end]
            self.airways = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "labels",
                                                    "*.npz")))[:end]
            self.lungs = sorted(glob(os.path.join(processed_data_folder,
                                                   mode,
                                                   "lungs",
                                                   "*.npz")))[:end]
        else:
            self.cts = sorted(glob(os.path.join(processed_data_folder,
                                                mode,
                                                "images",
                                                "*.npz")))
            self.airways = sorted(glob(os.path.join(processed_data_folder,
                                                    mode,
                                                    "labels",
                                                    "*.npz")))
            self.lungs = sorted(glob(os.path.join(processed_data_folder,
                                                   mode,
                                                   "lungs",
                                                   "*.npz")))
        self.transform = transform(**kwargs) if transform is not None else None
        assert len(self.cts) == len(self.lungs)
        assert len(self.cts) == len(self.airways)

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, idx: int):
        '''
        Carregar, transformar e retornar o item 'i' do dataset
        '''
        ct_path = self.cts[idx]
        ct_lungs_path = self.lungs[idx]
        ct_airways_path = self.airways[idx]

        # Ler os .npz salvos no pre processamento
        # print(f'Reading {ct_path} and {ct_labels_path}.......')
        image_npz = np.load(ct_path)
        lung_npz = np.load(ct_lungs_path)
        airway_npz = np.load(ct_airways_path)

        ct = image_npz['arr_0']
        lung = lung_npz['arr_0']
        airway = airway_npz['arr_0']
        ct = torch.tensor(ct).to(torch.float32)
        ct = ct.unsqueeze(0)
        airway = torch.tensor(airway.astype(float)).to(torch.float32).unsqueeze(0)
        lung = torch.tensor(lung).to(torch.float32).unsqueeze(0)

        # Se uma função de transformada foi passada para o dataset, aplicá-la
        if self.transform is not None:
            lung = self.transform(lung)
        # Retornar a imagem e metadados
        return ct, airway, lung