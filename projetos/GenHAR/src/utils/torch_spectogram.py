import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

###Definir o transformador de espectrograma usando torchaudio
def create_spectrogram_torchaudio(signal, n_fft=40, hop_length=20, win_length=40, power=2.0):
    """
    Cria um espectrograma para um sinal específico usando torchaudio.
    """
    signal_tensor = torch.tensor(signal).float()
    
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        power=power
    )
    
    spectrogram = spectrogram_transform(signal_tensor)
    return spectrogram

# Função para gerar e salvar todos os espectrogramas com categorias e nomes de categorias
def save_spectrograms(output_file, X, y, category_names):
    """
    Gera e salva todos os espectrogramas para cada classe no dataset.
    """
    unique_classes = np.unique(y)
    spectrograms_dict = {}
    
    for class_label in unique_classes:
        # Selecionar todas as amostras dessa classe
        class_indices = np.where(y == class_label)[0]
        
        spectrograms_list = []
        
        for index in class_indices:
            signal = X[index]
            spectrogram = create_spectrogram_torchaudio(signal)
            
            # Converter o espectrograma para um array NumPy
            spectrogram_np = spectrogram.numpy()
            spectrograms_list.append(spectrogram_np)
        
        # Salvar espectrogramas no dicionário
        spectrograms_dict[f'class_{class_label}'] = np.array(spectrograms_list)
    
    # Salvar os dados no arquivo .npz
    np.savez(output_file, specs=spectrograms_dict, categories=unique_classes, category_names=category_names)
    print(f"Espectrogramas salvos no arquivo '{output_file}'.")

def load_spectrograms(input_file):
    """
    Carrega espectrogramas e informações relacionadas de um arquivo .npz.
    """
    data = np.load(input_file, allow_pickle=True)  # Permitir o carregamento de objetos
    spectrograms_dict = data['specs'].item()  # Carregar o dicionário de espectrogramas
    categories = data['categories']
    category_names = data['category_names']
    return spectrograms_dict, categories, category_names

def plot_spectrograms_subplot(spectrograms_dict, category_names, num_spectrograms=3):
    """
    Plota espectrogramas em subplots, mostrando uma quantidade específica por categoria.
    """
    unique_classes = list(spectrograms_dict.keys())
    num_classes = len(unique_classes)
    
    fig, axes = plt.subplots(num_classes, num_spectrograms, figsize=(15, num_classes * 5))
    
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=0)  
    
    for class_idx, class_label in enumerate(unique_classes):
        spectrograms = spectrograms_dict[class_label]
        
        for i in range(num_spectrograms):
            ax = axes[class_idx, i]
            ax.imshow(np.log2(spectrograms[i] + 1e-9), cmap='viridis', aspect='auto', origin='lower')
            ax.set_title(f'{category_names[class_idx]} - Ex {i}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.title="spectograms using torch library"
    plt.show()
    
    return fig

