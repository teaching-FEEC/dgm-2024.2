import torch
from torch.utils.data import Dataset, DataLoader
import random
random.seed(5)
import matplotlib.pyplot as plt
import numpy as np
import os

from datasets import lungCTData
from model import Generator
from metrics import calculate_fid, FeatureExtractor, get_features
from metrics import my_ssim, crop_center_array
from utils import plt_save_example_synth_during_test, save_quantitative_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


trained_gen_dir = './model_reg_center_thr_25k/'
trained_gen_path = trained_gen_dir+'models/model_reg_center_thr_25k_gen_trained.pt'
dir_to_save_gen_imgs = trained_gen_dir+'generated_imgs/'
path_to_save_metrics = trained_gen_dir+'quantitative_metrics.json'
os.makedirs(dir_to_save_gen_imgs, exist_ok=True)

processed_data_folder = '/mnt/shared/ctdata_thr25'
start_point_test_data = 20000
end_point_test_data = 21000
batch_size = 10

dataset_test = lungCTData(processed_data_folder=processed_data_folder,
                          mode='test',
                          start=start_point_test_data,
                          end=end_point_test_data)

# Define gerador
gen = Generator()
gen.load_state_dict(torch.load(trained_gen_path, weights_only=True))
gen.to(device)
gen.eval()
data_loader_test = DataLoader(dataset_test,
                              batch_size=batch_size,
                              shuffle=False)

# Define a rede Inception V3
model_inception = torch.hub.load('pytorch/vision:v0.10.0',
                                 'inception_v3',
                                 pretrained=True)
model_inception.eval()
# Define modelo para feature extraction
feature_extractor = FeatureExtractor(model_inception)

# Gera dados sintéticos
fake_data_imgs = np.empty((len(dataset_test), 512, 512))
real_data_imgs = np.empty((len(dataset_test), 512, 512))
fake_data_features = np.empty((len(dataset_test), 2048))
real_data_features = np.empty((len(dataset_test), 2048))
counter = 0
with torch.no_grad():
    for batch in data_loader_test:
        input_img_batch = batch[0]
        input_mask_batch = batch[1]

        input_img = input_img_batch.to(device)
        input_mask = input_mask_batch.to(device)

        gen_img = gen(input_mask)
        fake_data_imgs[counter*batch_size:(counter+1)*batch_size, :, :] = np.squeeze(gen_img.detach().cpu().numpy())
        real_data_imgs[counter*batch_size:(counter+1)*batch_size, :, :] = np.squeeze(input_img.detach().cpu().numpy())

        features_fake = get_features(feature_extractor, gen_img, choose_transform=2, device=device)
        features_real = get_features(feature_extractor, input_img, choose_transform=2, device=device)
        fake_data_features[counter*batch_size:(counter+1)*batch_size, :] = features_fake
        real_data_features[counter*batch_size:(counter+1)*batch_size, :] = features_real
        counter = counter+1

        plt_save_example_synth_during_test(input_img_ref=np.squeeze(input_img.detach().cpu().numpy())[0, :, :],
                                           gen_img_ref=np.squeeze(gen_img.detach().cpu().numpy())[0, :, :],
                                           save_dir=dir_to_save_gen_imgs,
                                           img_idx=counter)

# Obtém as distribuições para os dados reais e sintéticos
mu1, sigma1 = np.mean(np.squeeze(real_data_features), axis=0), np.cov(np.squeeze(real_data_features))
mu2, sigma2 = np.mean(np.squeeze(fake_data_features), axis=0), np.cov(np.squeeze(fake_data_features))
# Calcula o FID entre os dados reais e sintéticos
fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
print(f"FID: {fid_value:.2f}")

# Calcula SSIM entre dados reais e sintéticos
ssim_complete = np.zeros(len(dataset_test))
luminance_complete = np.zeros(len(dataset_test))
contraste_complete = np.zeros(len(dataset_test))
struct_similarity_complete = np.zeros(len(dataset_test))
for i in range(len(dataset_test)):
    ssim_complete[i], luminance_complete[i], contraste_complete[i], struct_similarity_complete[i] = my_ssim(np.squeeze(real_data_imgs[i]), np.squeeze(fake_data_imgs[i]), 0.01, 0.03, 0.03)
print(np.mean(ssim_complete), np.std(ssim_complete))

# Calcula SSIM entre dados reais e sintéticos
ssim_center = np.zeros(len(dataset_test))
luminance_center = np.zeros(len(dataset_test))
contraste_center = np.zeros(len(dataset_test))
struct_similarity_center = np.zeros(len(dataset_test))
for i in range(len(dataset_test)):
    ssim_center[i], luminance_center[i], contraste_center[i], struct_similarity_center[i] = my_ssim(np.squeeze(crop_center_array(real_data_imgs[i], 256, 256)), np.squeeze(crop_center_array(fake_data_imgs[i], 256, 256)), 0.01, 0.03, 0.03)
print(np.mean(ssim_center), np.std(ssim_center))

save_quantitative_results(fid=fid_value,
                          ssim_complete=ssim_complete,
                          luminance_complete=luminance_complete,
                          contrast_complete=contraste_complete,
                          struct_sim_complete=struct_similarity_complete,
                          ssim_center=ssim_center,
                          luminance_center=luminance_center,
                          contrast_center=contraste_center,
                          struct_sim_center=struct_similarity_center,
                          save_path=path_to_save_metrics)
