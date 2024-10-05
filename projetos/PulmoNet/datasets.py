from torch.utils.data import Dataset
import torch
import numpy as np
from glob import glob
import SimpleITK as sitk
from lungmask import LMInferer


class rawCTData(Dataset):
    def __init__(self, mode: str, transform: Optional[Callable] = None):
        super().__init__()
        if mode == 'train':
            self.cts = sorted(glob(os.path.join(RAW_DATA_FOLDER, mode, "imagesTr", "*.nii.gz")))
            self.labels = sorted(glob(os.path.join(RAW_DATA_FOLDER, mode, "labelsTr", "*.nii.gz")))
        else:
            self.cts = sorted(glob(os.path.join(RAW_DATA_FOLDER, mode, "*.nii.gz")))
        self.transform = transform
        self.inferer = LMInferer()

    def __len__(self):
        return len(self.cts)
    
    def __getitem__(self, idx: int):
        '''
        Carregar, transformar e retornar o item 'i' do dataset
        Cada aquisição envolve 4 sequências presentes na dimensão de canais da imagem
        As sequências mapeam com a string RawMRIDataset.SEQUENCE
        '''
        ct_path = self.cts[idx]
        ct_labels_path = self.labels[idx]

        # Ler imagem usando a biblioteca SimpleITK, o objeto image contêm também metadados
        print(f'Reading {ct_path} and {ct_labels_path}.......')
        image = sitk.ReadImage(ct_path)
        label = sitk.ReadImage(ct_labels_path)

        # Converter imagem para array numpy na variável 'ct'
        print("Converting to array")
        ct = sitk.GetArrayFromImage(image)
        ct_label = sitk.GetArrayFromImage(label)
        ct_lung = self.inferer.apply(ct)
        ct_lung[ct_lung>1] = 1

        # Se uma função de transformada foi passada para o dataset, aplicá-la
        if self.transform is not None:
            ct = self.transform(ct)
        print(ct.shape)
        # Retornar a imagem e metadados
        return ct, ct_label, ct_lung


class lungCTData(Dataset):
    def __init__(self, mode: str, qntty: Optional[int] = None, transform: Optional[Callable] = None):
        super().__init__()
        if qntty is not None:
            self.cts = sorted(glob(os.path.join(PROCESSED_DATA_FOLDER, mode, "imagesTr", "*.npz")))[:qntty]
            self.labels = sorted(glob(os.path.join(PROCESSED_DATA_FOLDER, mode, "lungsTr", "*.npz")))[:qntty]
        else:
            self.cts = sorted(glob(os.path.join(PROCESSED_DATA_FOLDER, mode, "imagesTr", "*.npz")))
            self.labels = sorted(glob(os.path.join(PROCESSED_DATA_FOLDER, mode, "lungsTr", "*.npz")))
        self.transform = transform
        assert len(self.cts) == len(self.labels)

    def __len__(self):
        return len(self.cts)
    
    def __getitem__(self, idx: int):
        '''
        Carregar, transformar e retornar o item 'i' do dataset
        Cada aquisição envolve 4 sequências presentes na dimensão de canais da imagem
        As sequências mapeam com a string RawMRIDataset.SEQUENCE
        '''
        ct_path = self.cts[idx]
        ct_labels_path = self.labels[idx]

        # Ler imagem usando a biblioteca SimpleITK, o objeto image contêm também metadados
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
            ct = self.transform(ct)
        # Retornar a imagem e metadados
        return ct, lung