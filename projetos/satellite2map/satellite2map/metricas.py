import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

"""
Funções de avaliação de imagens RGB comresolução 256x256,  transferidas em Batches de 32 imagens.
"""

# Função para MSE
def mse_metric(batch_pred, batch_true):
    """
    Calcula o MSE para um batch de imagens.

    :param batch_pred: Tensor de previsão com forma (32, 3, 256, 256).
    :param batch_true: Tensor de referência com forma (32, 3, 256, 256).
    :return: MSE médio para o batch.
    """
    mse = F.mse_loss(batch_pred, batch_true, reduction='mean')
    return mse.item()

# Função para PSNR
def psnr_metric(batch_pred, batch_true):
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
def ssim_metric(batch_pred, batch_true):
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
        pred_img = batch_pred[i].permute(1, 2, 0).cpu().numpy()
        true_img = batch_true[i].permute(1, 2, 0).cpu().numpy()

        # Calcula o SSIM por canal e tira a média.
        ssim_value = ssim(pred_img, true_img, multichannel=True, data_range=pred_img.max() - pred_img.min())
        ssim_values.append(ssim_value)

    return np.mean(ssim_values)
