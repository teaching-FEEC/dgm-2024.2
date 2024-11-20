def normalize(s, mel_means, mel_stds):
    """
    Normaliza um espectrograma usando médias e desvios padrões fornecidos.
    """
    norm_Y = (s - mel_means) / (3.0 * mel_stds)
    return np.clip(norm_Y, -1.0, 1.0)

from skimage.transform import resize


def normalize_1(mel_spectrogram, mel_means, mel_stds):
    """
    Normaliza o espectrograma Mel usando médias e desvios padrões fornecidos.
    
    Args:
        mel_spectrogram (ndarray): Espectrograma Mel a ser normalizado.
        mel_means (float): Média usada para normalização.
        mel_stds (float): Desvio padrão usado para normalização.
    
    Returns:
        ndarray: Espectrograma Mel normalizado.
    """
    return (mel_spectrogram - mel_means) / mel_stds


import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt


def load_melspecs_from_df(row_values, label, mel_means, mel_stds):
    """
    Processa uma linha do DataFrame para gerar um espectrograma Mel e normaliza-lo.

    Args:
        row_values (np.ndarray): Valores da linha do DataFrame, representando um espectrograma Mel.
        label (int): Rótulo da categoria.
        mel_means (float): Média para normalização.
        mel_stds (float): Desvio padrão para normalização.

    Returns:
        tuple: (espectrograma normalizado, índice da categoria) ou (None, None) se não for possível processar.
    """
    try:
        # Converter a linha para um array 2D
        mel_spectrogram = row_values.reshape(-1, 360 // 128)

        # Converter para dB
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalizar
        norm_mel =  normalize(mel_spectrogram_db, mel_means, mel_stds)
        
        return norm_mel, label
    except Exception as e:
        print(f"Erro ao processar o espectrograma: {e}")
        return None, None
    
import numpy as np
import joblib

import numpy as np
import joblib

def process_mel_spectrograms(data, labels,categories_names, output_file):
    print("Gerando espectrogramas Mel normalizados...")

    # Definir médias e desvios padrões para normalização
    mel_means = np.mean(data.values.flatten())
    mel_stds = np.std(data.values.flatten())


    # Processar os dados em paralelo
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(load_melspecs_from_df)(row.values, label, mel_means, mel_stds) 
        for (_, row), label in zip(data.iterrows(), labels)
    )

    # Separar espectrogramas e categorias
    specs = [spec for spec, _ in results if spec is not None]
    categories = [category for _, category in results if category is not None]

    # Salvar os dados em um arquivo npz
    print(f"Salvando dados em {output_file}")

    np.savez(output_file, specs=specs, categories=categories, category_names=categories_names,
             mean=mel_means, std=mel_stds)

    print("Concluído!")

# Exemplo de chamada da função
# process_mel_spectrograms(data, labels, 'dataset_name', 'mel_specs_dataset_name_output.npz')

# Função para pré-processar o espectrograma
def preprocess_spectrogram(spectrogram, target_shape):
    """
    Redimensiona o espectrograma para o formato desejado.

    Args:
        spectrogram (ndarray): Espectrograma a ser redimensionado.
        target_shape (tuple): Formato desejado para a entrada (altura, largura).
    
    Returns:
        ndarray: Espectrograma redimensionado.
    """
    return resize(spectrogram, target_shape, mode='reflect', anti_aliasing=True)

def plot_random_images_per_category(specs, categories, category_names, num_images=3, input_shape=(128, 128)):
    """
    Plota um número especificado de imagens de espectrogramas Mel escolhidas aleatoriamente para cada categoria.

    Args:
        specs (ndarray): Lista de espectrogramas Mel.
        categories (ndarray): Lista de categorias correspondentes.
        category_names (list): Lista de nomes das categorias.
        num_images (int): Número de imagens a serem exibidas por categoria.
        input_shape (tuple): Tamanho desejado da entrada (altura, largura).
    """
    unique_categories = np.unique(categories)
    figs=[]
    for cat_idx in unique_categories:
        # Filtrar espectrogramas para a categoria
        cat_specs = [spec for spec, cat in zip(specs, categories) if cat == cat_idx]

        if len(cat_specs) < num_images:
            print(f"Categoria {category_names[cat_idx]} tem menos de {num_images} imagens!")
            num_images = len(cat_specs)

        random_indices = np.random.choice(len(cat_specs), size=num_images, replace=False)
        selected_specs = [cat_specs[i] for i in random_indices]

        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        fig.suptitle(f" Espectograma Categoria: {category_names[cat_idx]}")

        for i in range(num_images):
            resized_spec = preprocess_spectrogram(selected_specs[i], target_shape=input_shape)
            #print(resized_spec.shape,selected_specs[i].shape)
            axes[i].imshow(resized_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[i].set_title(f"Imagem {i+1}")
            axes[i].axis('off')
        figs.append(fig)
        plt.show()
    return figs