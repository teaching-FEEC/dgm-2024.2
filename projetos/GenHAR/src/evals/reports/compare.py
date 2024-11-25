import os
import matplotlib.pyplot as plt
from PIL import Image

def load_images_in_grid(datasets, models, result_path,type_plot, save_folder,save_file_name):
    image_paths = []
    
    # Gerar os caminhos das imagens baseado nos datasets e modelos
    for dataset in datasets:
        for model in models:
            image_name = f'{dataset}/_{model}___{type_plot}.jpg'
            image_path = os.path.join(result_path, image_name)
            if os.path.exists(image_path):
                image_paths.append((image_path, dataset, model))  
            else:
                print(f'Warning: {image_name} not found in {result_path}')
    
    # Calcular o número total de imagens
    total_images = len(image_paths)
    
    if total_images == 0:
        print("No images found.")
        return
    
    # Calcular o grid (número de linhas e colunas) com base no número de datasets e modelos
    cols = len(datasets)  # Número de colunas será o número de datasets
    rows = len(models)    # Número de linhas será o número de modelos

    # Criar a figura
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 7))  # Aumentar o tamanho da figura

    # Remover qualquer espaçamento extra entre os subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Adicionar título superior (nomes dos datasets) na parte superior da figura
    for col in range(cols):
        axs[0, col].set_title(datasets[col], fontsize=18, pad=5)  # Aumentar o tamanho da fonte

    # Adicionar texto dos modelos nas filas
    for row in range(rows):
        axs[row, 0].text(-0.2, 0.5, models[row], fontsize=18, ha='right', va='center', transform=axs[row, 0].transAxes)  # Ajustar a posição e o tamanho do texto

    # Iterar sobre os caminhos das imagens e exibir cada uma no subplot correspondente
    for i, (image_path, dataset, model) in enumerate(image_paths):
        # Carregar a imagem
        img = Image.open(image_path)

        # Calcular a posição da imagem na grade
        row = models.index(model)
        col = datasets.index(dataset)

        # Exibir a imagem no subplot correspondente
        axs[row, col].imshow(img)
        axs[row, col].axis('off')  # Remover os eixos

    # Caso haja subplots a mais que não são usados, removê-los
    for j in range(total_images, rows * cols):
        fig.delaxes(axs.flatten()[j])

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f'{save_file_name}_{type_plot}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  
    

datasets_=["WISDM",  "MotionSense",    "UCI" ,     "KuHar",
    "RealWorld_thigh", "RealWorld_upperarm", "RealWorld_waist"]

# por dataset
result_path = 'reports/results_all/images'  # Caminho onde as imagens estão armazenadas

for dataset in datasets_:
    datasets = [f'{dataset}_None',f'{dataset}_fft']  # Exemplo de datasets
    models = ['uncond_diffusion_unet1d','cond_diffusion_unet1d','Doppelgangerger', 'timeganpt']  # Exemplo de modelos

    # Chamar a função para carregar as imagens e exibi-las em uma matriz ajustável
    load_images_in_grid(datasets, models, result_path=result_path, type_plot='tsne',save_folder='reports/results_all/images/compare',save_file_name=f'{dataset}_comp')
    load_images_in_grid(datasets, models, result_path=result_path, type_plot='distribution',save_folder='reports/results_all/images/compare',save_file_name=f'{dataset}_comp')

datasets_None = [
    "WISDM_None", 
    "MotionSense_None", 
    "UCI_None", 
    "KuHar_None",
    "RealWorld_thigh_None", 
    "RealWorld_upperarm_None", 
    "RealWorld_waist_None"
]

load_images_in_grid(datasets_None, models, result_path=result_path, type_plot='tsne',save_folder='reports/results_all/images/compare',save_file_name=f'all_comp_none')
load_images_in_grid(datasets_None, models, result_path=result_path, type_plot='distribution',save_folder='reports/results_all/images/compare',save_file_name=f'all_comp_none')

datasets_fft = [
    "WISDM_fft", 
    "MotionSense_fft", 
    "UCI_fft", 
    "KuHar_fft",
    "RealWorld_thigh_fft", 
    "RealWorld_upperarm_fft", 
    "RealWorld_waist_fft"
]

load_images_in_grid(datasets_fft, models, result_path=result_path, type_plot='tsne',save_folder='reports/results_all/images/compare',save_file_name=f'all_comp_fft')
load_images_in_grid(datasets_fft, models, result_path=result_path, type_plot='distribution',save_folder='reports/results_all/images/compare',save_file_name=f'all_comp_fft')

datasets_none_fft = [
    "WISDM_None", "WISDM_fft",
    "MotionSense_None", "MotionSense_fft",
    "UCI_None", "UCI_fft",
    "KuHar_None", "KuHar_fft",
    "RealWorld_thigh_None", "RealWorld_thigh_fft",
    "RealWorld_upperarm_None", "RealWorld_upperarm_fft",
    "RealWorld_waist_None", "RealWorld_waist_fft"
]

load_images_in_grid(datasets_none_fft, models, result_path=result_path, type_plot='tsne',save_folder='reports/results_all/images/compare',save_file_name=f'all_comp')
load_images_in_grid(datasets_none_fft, models, result_path=result_path, type_plot='distribution',save_folder='reports/results_all/images/compare',save_file_name=f'all_comp')

## python src/eval/reports/compare.py