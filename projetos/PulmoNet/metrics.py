# metrics.py
import torch
from torchvision import transforms
from scipy.linalg import sqrtm
import numpy as np


# FID ---------------------------------------------
# Referencia: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def calculate_fid(mu1, sigma1, mu2, sigma2):
    # Calculate the squared difference between means
    diff = mu1 - mu2
    diff_squared = np.sum(diff**2)

    # Calculate the product of the covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate the FID score
    fid = diff_squared + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid


# Define uma classe que herda do Inception V3 e modifica sua última camada
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model_inceptionv3):
        super(FeatureExtractor, self).__init__()
        self.model_inceptionv3 = model_inceptionv3
        # Remove a última camada (fully connected) usada pelo classificador
        self.model_inceptionv3.fc = torch.nn.Identity()

    def forward(self, x):
        x = self.model_inceptionv3(x)
        return x


# Define uma função que encontra o meio de um tensor e corta a matriz conforme as dimensões de entrada
def crop_center_tensor(tensor, cropx, cropy):
    _, _, y, x = tensor.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return tensor[:, :, starty:starty+cropy, startx:startx+cropx]


def crop_center_array(array, cropx, cropy):
    y, x = array.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return array[starty:starty+cropy, startx:startx+cropx]


# Obtém as features do InceptionV3
def get_features(feature_extractor, input_tensor, choose_transform=1, device='cpu'):
    # Adiciona canal de dimensão já que está em escala cinza
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Now shape is (1, 1, 512, 512)

    # Define o pré-processamento
    if (choose_transform == 1):
        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize to 299x299
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Repeat grayscale to 3 channels if necessary
          ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Repeat grayscale to 3 channels if necessary
          ])

    # Aplica a transformação
    if (choose_transform == 1):
        input_batch = transform(input_tensor.float())  # shape = 1, 3, 299, 299
    else:
        cropped_tensor = crop_center_tensor(input_tensor.float(), 299, 299)
        input_batch = transform(cropped_tensor.float())  # shape = 1, 3, 299, 299

    # move the input and model to GPU for speed if available
    input_batch = input_batch.to(device)
    feature_extractor.to(device)

    # Obtem as features para os dados reais
    features = feature_extractor(input_batch).detach().cpu().numpy()  # torch.Size([1, 2048])

    return features


# SSIM ------------------------------------------------------------------------

# Calcula luminância
def luminance(img1, img2, C1):
    l = (2 * np.mean(img1) * np.mean(img2) + C1) / (np.mean(img1)**2 + np.mean(img2)**2 + C1)
    return l


# Calcula a distorção de contraste
def contrast(img1, img2, C2):
    c = (2 * np.std(img1) * np.std(img2) + C2) / (np.std(img1)**2 + np.std(img2)**2 + C2)
    return c


def cross_covariance(image1, image2):
    """Calculates the cross-covariance between two images.

    Args:
      image1: The first image as a NumPy array.
      image2: The second image as a NumPy array.

    Returns:
      The cross-covariance matrix.
    """

    # Ensure images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape.")

    # Flatten the images
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # Calculate the cross-covariance
    cross_cov = np.cov(image1_flat, image2_flat, bias=True)[0, 1]

    return cross_cov


# Calcula a perda de correlação estrutural
def structure_similarity(img1, img2, C3):
    s = (cross_covariance(img1, img2) + C3) / (np.std(img1) * np.std(img2) + C3)
    return s


# Calcula o SSIM entre duas imagens:
def my_ssim(img1, img2, C1, C2, C3):
    l = luminance(img1, img2, C1)
    c = contrast(img1, img2, C2)
    s = structure_similarity(img1, img2, C3)
    # print(f"luminance: {l:.2f}")
    # print(f"contrast: {c:.2f}")
    # print(f"structure_similarity: {s:.2f}")
    return l * c * s, l, c, s