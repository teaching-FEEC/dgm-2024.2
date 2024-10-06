import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from datasets import rawCTData, lungCTData

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

def preprocessing_for_lung(raw_data_folder, processed_data_folder):
    transform = MinMaxNormalize()
    file_number = 0
    for mode in ['train']: #, 'val']:
        if mode == 'train':
            os.makedirs(os.path.join(processed_data_folder, mode, 'imagesTr'), exist_ok=True)
            os.makedirs(os.path.join(processed_data_folder, mode, 'labelsTr'), exist_ok=True)
            os.makedirs(os.path.join(processed_data_folder, mode, 'lungsTr'), exist_ok=True)
        else:
            os.makedirs(os.path.join(processed_data_folder, mode), exist_ok=True)
        raw_data = rawCTData(raw_data_folder=raw_data_folder, mode=mode, transform=transform)

        for volume, _, lung in tqdm(raw_data):
            print(volume.shape, lung.shape)
            for ct_slice, ct_lung in zip(volume, lung):
                # Só estou processando fatias que tenham pulmão
                if np.sum(ct_lung) > 100:
                    img_filename = os.path.join(processed_data_folder, mode, 'imagesTr', f"{file_number}.npz")
                    #label_filename = os.path.join(processed_data_folder, mode, 'labelsTr',f"{file_number}.npz")
                    lung_filename = os.path.join(processed_data_folder, mode, 'lungsTr',f"{file_number}.npz")
                    file_number += 1
                    np.savez_compressed(img_filename, ct_slice)
                    #np.savez_compressed(label_filename, ct_slice)
                    np.savez_compressed(lung_filename, ct_lung)

def test_processed_data(processed_data_folder):
    ct_path = sorted(glob(os.path.join(processed_data_folder, 'train', 'imagesTr', '*.npz')))
    lung_path = sorted(glob(os.path.join(processed_data_folder, 'train', 'lungsTr', '*.npz')))
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

def test_lung_dataset(processed_data_folder):
    data = lungCTData(processed_data_folder=processed_data_folder,mode='train')
    x,y = random.choice(data)
    assert x.shape == y.shape
    assert np.max(y) == 1
    assert np.max(x) <= 1
    assert np.min(x) >= 0

    plot_img_label(x,y)


def clean_directory(dir_path):
    #credits: Gabriel Dias (g172441@dac.unicamp.br), Mateus Oliveira (m203656@dac.unicamp.br)
    # from Github repo: https://github.com/MICLab-Unicamp/Spectro-ViT
    for file_name in os.listdir(dir_path):
        file_absolute_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_absolute_path):
            os.remove(file_absolute_path)
        elif os.path.isdir(file_absolute_path):
            clean_directory(file_absolute_path)
            os.rmdir(file_absolute_path)

def plt_save_example_synth_img(input_img_ref, input_mask_ref, gen_img_ref, disc_ans, epoch, save_dir): 
        fig, ax = plt.subplots(1,3,figsize=(18,4))
        ax.flat[0].imshow(input_img_ref,cmap='gray')
        ax.flat[1].imshow(input_mask_ref,cmap='gray')
        ax.flat[2].imshow(gen_img_ref,cmap='gray')
        ax.flat[0].set_title('Original')
        ax.flat[1].set_title('Mask')
        ax.flat[2].set_title('Generated: Disc answer: '+str(disc_ans))
        plt.savefig(save_dir+'example_generated_epoch_'+str(epoch)+'.png')