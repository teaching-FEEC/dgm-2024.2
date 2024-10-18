import torch
from torch.utils.data import Dataset, DataLoader
import random
random.seed(5)
import matplotlib.pyplot as plt
import numpy as np
import os

from datasets import lungCTData
from model import Generator
from metrics import my_fid_pipeline, my_ssim_pipeline
from utils import plt_save_example_synth_during_test, save_quantitative_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definição dos caminhos para localização do modelo e onde dados serão armazenados
trained_gen_dir = './model_reg_center_thr_25k/'
trained_gen_path = trained_gen_dir+'models/model_reg_center_thr_25k_gen_trained.pt'
dir_to_save_gen_imgs = trained_gen_dir+'generated_imgs/'
path_to_save_metrics = trained_gen_dir+'quantitative_metrics.json'
os.makedirs(dir_to_save_gen_imgs, exist_ok=True)

processed_data_folder = '/mnt/shared/ctdata_thr25'

# Parâmetros definidos para a avaliação dos resultados
start_point_test_data = 20000
end_point_test_data = 21000
batch_size = 10

# Obtenção dos dados de teste
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

# Gera dados sintéticos para análise qualitativa (visual e subjetiva)
counter = 0
with torch.no_grad():
    for batch in data_loader_test:
        input_img_batch = batch[0]
        input_mask_batch = batch[1]

        input_img = input_img_batch.to(device)
        input_mask = input_mask_batch.to(device)

        gen_img = gen(input_mask)
        counter=counter+1

        plt_save_example_synth_during_test(input_img_ref=np.squeeze(input_img.detach().cpu().numpy())[0, :, :],
                                           gen_img_ref=np.squeeze(gen_img.detach().cpu().numpy())[0, :, :],
                                           save_dir=dir_to_save_gen_imgs,
                                           img_idx=counter)

# Calcula FID
fid_value = my_fid_pipeline(dataset_test, data_loader_test, device, gen, batch_size)

# Calcula SSIM entre dados reais e sintéticos
# Imagem completa
ssim_complete, luminance_complete, contraste_complete, struct_similarity_complete = my_ssim_pipeline(dataset_test, 
                                                                                                     data_loader_test, 
                                                                                                     device, gen, batch_size, 
                                                                                                     bComplete=True)
# Imagem parcial (apenas centro)
ssim_center, luminance_center, contraste_center, struct_similarity_center = my_ssim_pipeline(dataset_test, 
                                                                                             data_loader_test, 
                                                                                             device, 
                                                                                             gen, 
                                                                                             batch_size, 
                                                                                             bComplete=False)

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
