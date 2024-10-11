import tensorflow_hub as hub
import tensorflow as tf
import numpy as np 
import io
from PIL import Image
import cv2

##Nome da função: metricMUSIQ_for_batch
##Objetivo: Função que retorna o valor da média da métrica MUSIQ para um determinado 
# batch de imagens.
##Parâmetros: Um batch de imagens no formato (B,H,W,C)
##Retorno: MUSIQ médio para o batch de entrada
def metricMUSIQ_for_batch(batch):

    # Carrega o modelo MUSIQ
    model = hub.load('https://www.kaggle.com/models/google/musiq/TensorFlow2/spaq/1')
    predict_fn = model.signatures['serving_default']

    # Calcula o MUSIQ para cada imagem do batch
    for image in batch:
        # Pega os bytes da imagem
        image_pil = Image.fromarray(image) 
        image_byte_array = io.BytesIO()
        image_pil.save(image_byte_array, format='PNG')
        image_bytes = image_byte_array.getvalue()

        # Calcula o MUSIQ
        aesthetic_score = predict_fn(tf.constant(image_bytes))
        sum = aesthetic_score['output_0'].numpy()

    # Retorna o MUSIQ médio para o batch
    return sum/len(batch)


##Nome da função: metricPSNR_for_batch
##Objetivo: Função que retorna o valor da média da métrica PSNR para um determinado 
# batch de imagens.
##Parâmetros: Dois batchs de imagens no formato (B,H,W,C), um com as imagens originais e 
# o outro com as respectivas compressões
##Retorno: PSNR médio para o batch de entrada
def metricPSNR_for_batch(original, compressed): 
    sum = 0

    # Calcula o PSNR para cada imagem original e sua respectiva compressão
    for in_original, in_compressed in zip(original, compressed):
        # Calcula o MSE
        mse = np.mean((np.array(in_original, dtype=np.float32) - np.array(in_compressed, dtype=np.float32)) ** 2)
        if(mse == 0): 
            return 100
        max_pixel = 255.0
        # Calcula o PSNR
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
        sum+=psnr

    # Retorna o PSNR médio para o batch
    return sum/len(original)


##Nome da função: metricMSSSIM_for_batch
##Objetivo: Função que retorna o valor da média da métrica MS-SSIM para um determinado 
# batch de imagens.
##Parâmetros: Dois batchs de imagens no formato (B,H,W,C), um com as imagens originais e 
# o outro com as respectivas compressões
##Retorno: MS-SSIM médio para o batch de entrada
def metricMSSSIM_for_batch(original, compressed):
    # O tf.image.ssim_multiscale retorna um tensor com o valor de MS-SSIM para cada 
    # imagem do batch
    sum = np.sum(tf.image.ssim_multiscale(original, compressed, 255).numpy())

    # Retorna o MS-SSIM médio para o batch
    return sum / len(tf.image.ssim_multiscale(original, compressed, 255).numpy())


##Nome da função: metricFM_for_batch
##Objetivo: Função que retorna o valor da média da métrica Fature Matching (FM) para um determinado 
# batch de imagens.
##Parâmetros: Dois batchs de imagens no formato (B,H,W,C), um com as imagens originais e 
# o outro com as respectivas compressões
##Retorno: FM médio para o batch de entrada
def metricFM_for_batch(original, compressed):
    sum = 0
    # Cria um detector de características ORB
    orb = cv2.ORB_create()

    # Cria um objeto de correspondência de características
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Calcula o FM para cada imagem original e sua respectiva compressão
    for in_original, in_compressed in zip(original, compressed):

        # Converte as imagens para escala de cinza
        in_original = cv2.cvtColor(in_original, cv2.COLOR_BGR2GRAY)
        in_compressed = cv2.cvtColor(in_compressed, cv2.COLOR_BGR2GRAY)

        # Detecta os pontos chave e os descritores correspondentes
        keypoints1, descriptors1 = orb.detectAndCompute(in_original, None)
        keypoints2, descriptors2 = orb.detectAndCompute(in_compressed, None)

        # Encontra correspondências entre os descritores das 2 imagens
        matches = bf.match(descriptors1, descriptors2)
        # Ordena as correspondências da menor distância para maior
        matches = sorted(matches, key=lambda x: x.distance)

        # Calcula a razão entre o número de correspondências válidas e o número total de pontos chave
        total_keypoints = max(len(keypoints1), len(keypoints2))  
        match_ratio = len(matches) / total_keypoints if total_keypoints > 0 else 0
        sum+=match_ratio

    # Retorna o FM médio para o batch
    return sum/len(original)


# def metricIoU_for_batch(original, compressed):
#     sum = 0
#     for in_original, in_compressed in zip(original, compressed):
#         sum_iou = 0
#         in_original = cv2.cvtColor(in_original, cv2.COLOR_RGBA2GRAY)
#         in_compressed = cv2.cvtColor(in_compressed, cv2.COLOR_RGBA2GRAY)

#         for color in np.unique(in_original):
#             thresh1 = np.where(in_original == color, 255, 0).astype(np.uint8)
#             thresh2 = np.where(in_compressed == color, 255, 0).astype(np.uint8)

#             mask_and = cv2.bitwise_and(thresh1, thresh2)
#             mask_or = cv2.bitwise_or(thresh1, thresh2)

#             iou = np.sum(mask_and)/np.sum(mask_or) if np.sum(mask_or) !=0 else 0

#             sum_iou+=iou
        
#         sum += sum_iou/len(np.unique(in_original))

#     return sum/len(original)


##Nome da função: metricIoU_for_batch
##Objetivo: Função que retorna o valor da média da métrica IoU para um determinado 
# batch de imagens.
##Parâmetros: Dois batchs de imagens no formato (B,H,W,C), um com as imagens originais e 
# o outro com as respectivas compressões
##Retorno: IoU médio para o batch de entrada
def metricIoU_for_batch(original, compressed):
    sum = 0
    # Calcula o IoU para cada imagem original e sua respectiva compressão
    for in_original, in_compressed in zip(original, compressed):
        sum_iou = 0
        sum_pixel = 0

        # Converte as imagens para tons de cinza
        in_original = cv2.cvtColor(in_original, cv2.COLOR_RGBA2GRAY)
        in_compressed = cv2.cvtColor(in_compressed, cv2.COLOR_RGBA2GRAY)

        # Faz O IoU para cada segmentação (cor) da representação semântica
        for color in np.unique(in_original):
            # Aplica, nas duas imagens, a máscara correspondente ao tom tratado,
            # o que é igual a color vira branco e o que não é vira preto
            thresh1 = np.where(in_original == color, 255, 0).astype(np.uint8)
            thresh2 = np.where(in_compressed == color, 255, 0).astype(np.uint8)

            # Aplica uma interseção (and) e união (or) nas 2 imagens
            mask_and = cv2.bitwise_and(thresh1, thresh2)
            mask_or = cv2.bitwise_or(thresh1, thresh2)
            
            # Calcula o IoU, razão entre interseção e união
            iou = np.sum(mask_and)/np.sum(mask_or) if np.sum(mask_or) !=0 else 0
            sum_pixel+=np.sum(thresh1)
            
            # O IoU de cada segmentação é ponderado pela quantidade de pixels 1 da segmentação da
            # imagem original
            sum_iou+=iou*np.sum(thresh1)
        
        sum += sum_iou/sum_pixel
    # Retorna o IoU médio para o batch
    return sum/len(original)



##Nome da função: metricMUSIQ_for_image
##Objetivo: Função que retorna o valor da métrica MUSIQ para uma determinada
# imagens.
##Parâmetros: Uma imagem
##Retorno: MUSIQ para a imagem de entrada
def metricMUSIQ_for_image(image):

    model = hub.load('https://www.kaggle.com/models/google/musiq/TensorFlow2/spaq/1')
    predict_fn = model.signatures['serving_default']

    
    image_pil = Image.fromarray(image) 
    image_byte_array = io.BytesIO()
    image_pil.save(image_byte_array, format='PNG')
    image_bytes = image_byte_array.getvalue()

    aesthetic_score = predict_fn(tf.constant(image_bytes))

    return aesthetic_score['output_0'].numpy()


##Nome da função: metricPSNR_for_image
##Objetivo: Função que retorna o valor da média da métrica PSNR para uma determinada
# imagem
##Parâmetros: A imagem original e a compressão correspondente
##Retorno: PSNR para as imagens de entrada
def metricPSNR_for_image(original, compressed): 
    
    mse = np.mean((np.array(original, dtype=np.float32) - np.array(compressed, dtype=np.float32)) ** 2)
    if(mse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 

    return psnr


##Nome da função: metricFM_for_image
##Objetivo: Função que retorna o valor da métrica FM para uma determinada
# imagem
##Parâmetros: A imagem original e a compressão correspondente
##Retorno: FM para as imagens de entrada
def metricFM_for_image(original, compressed):
    
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    in_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    in_compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

    keypoints1, descriptors1 = orb.detectAndCompute(in_original, None)
    keypoints2, descriptors2 = orb.detectAndCompute(in_compressed, None)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    total_keypoints = max(len(keypoints1), len(keypoints2))  
    match_ratio = len(matches) / total_keypoints if total_keypoints > 0 else 0
    

    return match_ratio



##Nome da função: metricIoU_for_image
##Objetivo: Função que retorna o valor da métrica IoU para uma determinada
# imagem
##Parâmetros: A imagem original e a compressão correspondente
##Retorno: IoU para as imagens de entrada
def metricIoU_for_image(original, compressed):
    sum_iou = 0
    sum_pixel = 0
    in_original = cv2.cvtColor(original, cv2.COLOR_RGBA2GRAY)
    in_compressed = cv2.cvtColor(compressed, cv2.COLOR_RGBA2GRAY)

    for color in np.unique(in_original):
        thresh1 = np.where(in_original == color, 255, 0).astype(np.uint8)
        thresh2 = np.where(in_compressed == color, 255, 0).astype(np.uint8)

        mask_and = cv2.bitwise_and(thresh1, thresh2)
        mask_or = cv2.bitwise_or(thresh1, thresh2)

        iou = np.sum(mask_and)/np.sum(mask_or) if np.sum(mask_or) !=0 else 0
        sum_pixel+=np.sum(thresh1)

        sum_iou+=iou*np.sum(thresh1)
        

    return sum_iou/sum_pixel


##Nome da função: metricMSSSIM_for_image
##Objetivo: Função que retorna o valor da métrica MS-SSIM para uma determinada
# imagem
##Parâmetros: A imagem original e a compressão correspondente
##Retorno: MS-SSIM para as imagens de entrada
def metricMSSSIM_for_image(original, compressed):
    original = np.array(original)
    batch_original = original[np.newaxis]

    compressed = np.array(compressed)
    batch_compressed = compressed[np.newaxis]

    sum = np.sum(tf.image.ssim_multiscale(batch_original, batch_compressed, 255).numpy())
    return sum / len(tf.image.ssim_multiscale(batch_original, batch_compressed, 255).numpy())