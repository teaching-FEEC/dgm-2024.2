import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class MinMaxNormalize():
    '''
    Normaliza a imagem para o intervalo [0, 1]
    '''
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min())


def plot_img_label(img, label):
    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].imshow(img, cmap='gray')
    axarr[1].imshow(label, cmap='gray')
    plt.show()

def test_lung_segmentator(data):
    for volume, _, lung in data:
        return lung

def preprocessing_for_lung():
    transform = MinMaxNormalize()
    file_number = 0
    for mode in ['train']: #, 'val']:
        if mode == 'train':
            os.makedirs(os.path.join(PROCESSED_DATA_FOLDER, mode, 'imagesTr'), exist_ok=True)
            os.makedirs(os.path.join(PROCESSED_DATA_FOLDER, mode, 'labelsTr'), exist_ok=True)
            os.makedirs(os.path.join(PROCESSED_DATA_FOLDER, mode, 'lungsTr'), exist_ok=True)
        else:
            os.makedirs(os.path.join(PROCESSED_DATA_FOLDER, mode), exist_ok=True)
        raw_data = rawCTData(mode, transform=transform)

        for volume, _, lung in tqdm(raw_data):
            print(volume.shape, lung.shape)
            for ct_slice, ct_lung in zip(volume, lung):
                # Só estou processando fatias que tenham pulmão
                if np.sum(ct_lung) > 100:
                    img_filename = os.path.join(PROCESSED_DATA_FOLDER, mode, 'imagesTr', f"{file_number}.npz")
                    #label_filename = os.path.join(PROCESSED_DATA_FOLDER, mode, 'labelsTr',f"{file_number}.npz")
                    lung_filename = os.path.join(PROCESSED_DATA_FOLDER, mode, 'lungsTr',f"{file_number}.npz")
                    file_number += 1
                    np.savez_compressed(img_filename, ct_slice)
                    #np.savez_compressed(label_filename, ct_slice)
                    np.savez_compressed(lung_filename, ct_lung)

def test_processed_data():
    ct_path = sorted(glob(os.path.join(PROCESSED_DATA_FOLDER, 'train', 'imagesTr', '*.npz')))
    lung_path = sorted(glob(os.path.join(PROCESSED_DATA_FOLDER, 'train', 'lungsTr', '*.npz')))
    # confirmar quantidade de fatias
    assert len(ct_path) == len(lung_path)
    rnd_idx = random.randint(0, len(ct_path))
    ct_sample = np.load(ct_path[rnd_idx])['arr_0']
    lung_sample = np.load(lung_path[rnd_idx])['arr_0']
    assert ct_sample.shape == lung_sample.shape
    
    # comparação visual da ct e da segmentação
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(ct_sample, cmap='gray')
    axarr[1].imshow(lung_sample, cmap='gray')
    plt.show()

def test_lung_dataset():
    data = lungCTData('train')
    x,y = random.choice(data)
    assert x.shape == y.shape
    assert np.max(y) == 1
    assert np.max(x) <= 1
    assert np.min(x) >= 0

    plot_img_label(x,y)