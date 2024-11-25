# metrics.py
import torch
from torchvision import transforms
from scipy.linalg import sqrtm
import numpy as np


# FID ---------------------------------------------
def my_fid_pipeline(dataset_test, data_loader_test, device, gen, batch_size):
   # get Inception V3
  model_inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
  model_inception.eval()
  # defines model for feature extraction
  feature_extractor = FeatureExtractor(model_inception)

  fake_data_features = np.empty((len(dataset_test),2048))
  real_data_features = np.empty((len(dataset_test),2048))
  counter = 0
  with torch.no_grad():
      for batch in data_loader_test:
          input_img_batch = batch[0] # real image
          input_mask_batch = batch[1] # binary image
          
          input_img = input_img_batch.to(device)
          input_mask = input_mask_batch.to(device)
          
          # Generates synthetic data
          gen_img = gen(input_mask)

          # get features from last layer of InceptionV3
          features_fake = get_features(feature_extractor, gen_img, choose_transform=2, device=device)
          features_real = get_features(feature_extractor, input_img, choose_transform=2, device=device)
          fake_data_features[counter*batch_size:(counter+1)*batch_size,:] = features_fake
          real_data_features[counter*batch_size:(counter+1)*batch_size,:] = features_real
          counter=counter+1

  # get distribution of real and fake data
  mu1, sigma1 = np.mean(np.squeeze(real_data_features),axis=0), np.cov(np.squeeze(real_data_features))
  mu2, sigma2 = np.mean(np.squeeze(fake_data_features),axis=0), np.cov(np.squeeze(fake_data_features))

  # calculates FID
  fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)

  return fid_value

# Reference: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
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


# Define class that inherits Inception V3 and modifies its last layer
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model_inceptionv3):
        super(FeatureExtractor, self).__init__()
        self.model_inceptionv3 = model_inceptionv3
        # removes last layer (fully connected) used by the classifier
        self.model_inceptionv3.fc = torch.nn.Identity()

    def forward(self, x):
        x = self.model_inceptionv3(x)
        return x


# finds center of a 4D tensor and crops considering given dimensions
def crop_center_4D(input, cropx, cropy):
    _, _, x, y = input.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return input[:, :, startx:startx+cropx, starty:starty+cropy]

# finds center of a 2D tensor and crops considering given dimensions
def crop_center_2D(input, cropx, cropy):
    x, y = input.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return input[startx:startx+cropx, starty:starty+cropy]

# get features of InceptionV3
def get_features(feature_extractor, input_tensor, choose_transform=1, device='cpu'):
    # adds channel, since we are using grayscale images
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Now shape is (<batch_size>, 1, 512, 512)

    # pre-processing definition
    if (choose_transform == 1):
        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize to 299x299
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Repeat grayscale to 3 channels if necessary
          ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Repeat grayscale to 3 channels if necessary
          ])

    # transforms
    if (choose_transform == 1):
        input_batch = transform(input_tensor.float())  # shape = <batch_size>, 3, 299, 299
    else:
        cropped_tensor = crop_center_4D(input_tensor.float(), 299, 299)
        input_batch = transform(cropped_tensor.float())  # shape = <batch_size>, 3, 299, 299

    # move the input and model to GPU for speed if available
    input_batch = input_batch.to(device)
    feature_extractor.to(device)

    # extract features
    features = feature_extractor(input_batch).detach().cpu().numpy()  # torch.Size([<batch_size>, 2048])

    return features


# SSIM ------------------------------------------------------------------------
def my_ssim_pipeline(dataset_test, data_loader_test, device, gen, batch_size, bComplete):
   #generates synthetic data
  fake_data_imgs = np.empty((len(dataset_test),512,512))
  real_data_imgs = np.empty((len(dataset_test),512,512))
  counter = 0
  with torch.no_grad():
      for batch in data_loader_test:
          input_img_batch = batch[0] # real image
          input_mask_batch = batch[1] # mask
          
          input_img = input_img_batch.to(device)
          input_mask = input_mask_batch.to(device)
          
          gen_img = gen(input_mask)

          fake_data_imgs[counter*batch_size:(counter+1)*batch_size,:,:] = np.squeeze(gen_img.detach().cpu().numpy())
          real_data_imgs[counter*batch_size:(counter+1)*batch_size,:,:] = np.squeeze(input_img.detach().cpu().numpy())
          counter=counter+1
  
  # gets SSIM between real and fake data
  ssim = np.zeros(len(dataset_test))
  luminance = np.zeros(len(dataset_test))
  contraste = np.zeros(len(dataset_test))
  struct_similarity = np.zeros(len(dataset_test))

  # considering the full image (512 x 512)
  if (bComplete):
    for i in range(len(dataset_test)):
        ssim[i], luminance[i], contraste[i], struct_similarity[i] = my_ssim(np.squeeze(real_data_imgs[i]), np.squeeze(fake_data_imgs[i]), 0.01, 0.03, 0.03)
  # considering the center (256 x 256) -> focus on airways and lungs
  else:
    for i in range(len(dataset_test)):
        ssim[i], luminance[i], contraste[i], struct_similarity[i] = my_ssim(np.squeeze(crop_center_2D(real_data_imgs[i], 256, 256)), np.squeeze(crop_center_2D(fake_data_imgs[i], 256, 256)), 0.01, 0.03, 0.03)

  return ssim, luminance, contraste, struct_similarity

def my_ssim_pipeline(dataset_test, data_loader_test, device, gen, batch_size, bComplete):
   #generates synthetic data
  fake_data_imgs = np.empty((len(dataset_test),512,512))
  real_data_imgs = np.empty((len(dataset_test),512,512))
  counter = 0
  with torch.no_grad():
      for batch in data_loader_test:
          input_img_batch = batch[0] # real image
          input_mask_batch = batch[1] # mask
          
          input_img = input_img_batch.to(device)
          input_mask = input_mask_batch.to(device)
          
          gen_img = gen(input_mask)

          fake_data_imgs[counter*batch_size:(counter+1)*batch_size,:,:] = np.squeeze(gen_img.detach().cpu().numpy())
          real_data_imgs[counter*batch_size:(counter+1)*batch_size,:,:] = np.squeeze(input_img.detach().cpu().numpy())
          counter=counter+1
  
  # gets SSIM between real and fake data
  ssim = np.zeros(len(dataset_test))
  luminance = np.zeros(len(dataset_test))
  contraste = np.zeros(len(dataset_test))
  struct_similarity = np.zeros(len(dataset_test))

  # considering the full image (512 x 512)
  if (bComplete):
    for i in range(len(dataset_test)):
        ssim[i], luminance[i], contraste[i], struct_similarity[i] = my_ssim(np.squeeze(real_data_imgs[i]), np.squeeze(fake_data_imgs[i]), 0.01, 0.03, 0.03)
  # considering the center (256 x 256) -> focus on airways and lungs
  else:
    for i in range(len(dataset_test)):
        ssim[i], luminance[i], contraste[i], struct_similarity[i] = my_ssim(np.squeeze(crop_center_2D(real_data_imgs[i], 256, 256)), np.squeeze(crop_center_2D(fake_data_imgs[i], 256, 256)), 0.01, 0.03, 0.03)

  return ssim, luminance, contraste, struct_similarity

# gets luminance
def luminance(img1, img2, C1):
    l = (2 * np.mean(img1) * np.mean(img2) + C1) / (np.mean(img1)**2 + np.mean(img2)**2 + C1)
    return l


# gets contrast distortion
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


# gets loss of structure similarity
def structure_similarity(img1, img2, C3):
    s = (cross_covariance(img1, img2) + C3) / (np.std(img1) * np.std(img2) + C3)
    return s


# gets SSIM between two images
def my_ssim(img1, img2, C1, C2, C3):
    l = luminance(img1, img2, C1)
    c = contrast(img1, img2, C2)
    s = structure_similarity(img1, img2, C3)
    # print(f"luminance: {l:.2f}")
    # print(f"contrast: {c:.2f}")
    # print(f"structure_similarity: {s:.2f}")
    return l * c * s, l, c, s


# DICE -------------------------------------------------------------------
def dice_coefficient_score_calculation(pred, label, smooth=1e-5):
    '''
    From: https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/blob/main/evaluation/evaluation_atm_22.py
    '''
    pred = pred.flatten()
    label = label.flatten()
    intersection = np.sum(pred * label)
    dice_coefficient_score = round(((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth)) * 100, 2)
    return dice_coefficient_score