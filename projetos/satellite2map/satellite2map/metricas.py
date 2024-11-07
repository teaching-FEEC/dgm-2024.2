import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

"""
Funções de avaliação de imagens RGB comresolução 256x256,  transferidas em Batches de 32 imagens.
"""

# Função para MSE
def MSE(batch_pred, batch_true):
    """
    Calcula o MSE para um batch de imagens.

    :param batch_pred: Tensor de previsão com forma (32, 3, 256, 256).
    :param batch_true: Tensor de referência com forma (32, 3, 256, 256).
    :return: MSE médio para o batch.
    """
    mse = F.mse_loss(batch_pred, batch_true, reduction='mean')
    return mse.item()

# Função para PSNR
def PSNR(batch_pred, batch_true):
    """
    Calcula o PSNR para um batch de imagens.

    :param batch_pred: Tensor de previsão com forma (32, 3, 256, 256).
    :param batch_true: Tensor de referência com forma (32, 3, 256, 256).
    :return: PSNR médio para o batch.
    """
    mse = F.mse_loss(batch_pred, batch_true, reduction='mean')
    # Converta o mse para um tensor antes de aplicar torch.log10
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
    return psnr.item()

# Função para SSIM
def SSIM(batch_pred, batch_true):
    """
    Calcula o SSIM para um batch de imagens.

    :param batch_pred: Tensor de previsão com forma (32, 3, 256, 256).
    :param batch_true: Tensor de referência com forma (32, 3, 256, 256).
    :return: SSIM médio para o batch.
    """
    batch_size = batch_pred.size(0)
    ssim_values = []

    for i in range(batch_size):
        # Converte as imagens para numpy para calcular o SSIM com skimage.
        pred_img = batch_pred[i].detach().cpu().permute(1, 2, 0).numpy()
        true_img = batch_true[i].detach().cpu().permute(1, 2, 0).numpy()

        # Calcula o SSIM por canal e tira a média.
        ssim_value = ssim(pred_img, true_img, multichannel=True, channel_axis=2, data_range=pred_img.max() - pred_img.min())
        ssim_values.append(ssim_value)

    return np.mean(ssim_values)
\
def PA(batch_pred, batch_true, delta=5/255):
    """
    Calcula a acurácia de pixel para um batch de imagens.

    :param batch_pred: Tensor de previsão com forma (32, 3, 256, 256).
    :param batch_true: Tensor de referência com forma (32, 3, 256, 256).
    :param delta: Limiar de diferença para considerar um pixel correto.
    :return: Acurácia de pixel para o batch.
    """
    # scaling the images to [0, 1]
    batch_pred = batch_pred * 0.5 + 0.5
    batch_true = batch_true * 0.5 + 0.5

    # calculating the difference between the prediction and the ground truth
    diff = torch.abs(batch_pred - batch_true)

    # checking if the difference is less than delta
    correct_pixels = (diff < delta).sum()

    # if the difference is less than delta, the pixel is correct
    pixel_accuracy = correct_pixels.item() / torch.numel(batch_pred)

    return pixel_accuracy
