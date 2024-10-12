from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img
from PIL import Image
import numpy as np
import cv2
import os

def geracao(path_entrada, path_saida):

    # Define as transformações aplicadas pelo TF
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Cria o diretório de saída caso ele não exista
    if not os.path.exists(path_saida):
        os.makedirs(path_saida)

    # Verifica se a transformação já foi feita
    def is_duplicate(new_img, image_list):
        for img in image_list:
            if np.array_equal(new_img, img):
                return True
        return False

    # Pecorre todas as imagens do diretório de entrada
    for nome_arquivo in os.listdir(path_entrada):
        caminho_completo = os.path.join(path_entrada, nome_arquivo)

        if os.path.isfile(caminho_completo) and nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(caminho_completo)
            x = np.array(img)
            x = x.reshape((1,) + x.shape)  

            # Cria 2 imagens transformadas para cada imagem do diretório de entrada
            generated_images = []
            i = 0
            while i < 2: 
                for batch in datagen.flow(x, batch_size=1):
                    new_img = batch[0]
                    if not is_duplicate(new_img, generated_images): 
                        generated_images.append(new_img)
                        novo_nome = f"{os.path.splitext(nome_arquivo)[0]}_aug_{i}.png"
                        caminho_saida = os.path.join(path_saida, novo_nome)
                        save_img(caminho_saida, new_img)
                        i += 1
                    if i >= 2:
                        break

    
    # Percorre todo o diretório de saída
    for nome_arquivo in os.listdir(path_saida):
        caminho_completo = os.path.join(path_saida, nome_arquivo)
        
        # Cria uma imagem com transformação de brilho em cada imagem
        if os.path.isfile(caminho_completo) and nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(caminho_completo)

            if img is not None:
                img_brilho = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

                caminho_saida = os.path.join(path_saida, f'{nome_arquivo}_brilho.png')
                cv2.imwrite(caminho_saida, img_brilho)
