import torch
from torch.utils.data import DataLoader
import numpy as np
import os


from constants import *
from metrics import my_fid_pipeline, my_ssim_pipeline
from utils import read_yaml, plt_save_example_synth_during_test, save_quantitative_results


# Definição da função para avaliação dos resultados
def evaluate(gen, 
             trained_gen_dir,
             device,
             dataset_test,
             data_loader_test,
             batch_size = 10,
             bQualitativa=True, 
             bFID=True, 
             bSSIM=True):

    dir_to_save_gen_imgs = trained_gen_dir+'generated_imgs/'
    path_to_save_metrics = trained_gen_dir+'quantitative_metrics.json'
    os.makedirs(dir_to_save_gen_imgs, exist_ok=True)

    gen.eval()

    # Gera dados sintéticos para análise qualitativa (visual e subjetiva)
    if (bQualitativa is True):
        print('Generating examples for qualitative analysis...')
        counter = 0
        skip = 20
        with torch.no_grad():
            for batch in data_loader_test:
                if counter <= 20: 
                    if skip == 20:
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
                    else:
                        skip = skip - 1
                    if skip == 0:
                        skip=20

    # Calcula FID
    if (bFID is True):
        print('Calculating FID...')
        fid_value = my_fid_pipeline(dataset_test, data_loader_test, device, gen, batch_size)
    else:
        fid_value = np.nan

    # Calcula SSIM entre dados reais e sintéticos
    if (bSSIM is True):
        print('Calculating SSIM...')
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
    else:
        ssim_complete = np.nan
        luminance_complete = np.nan
        contraste_complete = np.nan
        struct_similarity_complete = np.nan
        ssim_center = np.nan
        luminance_center = np.nan
        contraste_center = np.nan
        struct_similarity_center = np.nan      

    # Salva os resultados
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
    print("Evaluation done!")
    
config_path = input("Enter path for YAML file with evaluation description: ")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = read_yaml(file=config_path)

##--------------------Definitions--------------------
#Generator model
model_gen_name = str(config['model']['name_model'])
use_best_version = bool(config['model']['use_best_version'])

#data
processed_data_folder = str(config['data'].get('processed_data_folder',
                                               '/mnt/shared/ctdata_thr25'))
dataset_type = str(config['data'].get('dataset',
                                    'lungCTData'))
start_point_test_data = config['data'].get('start_point_test_data',None)
end_point_test_data = config['data'].get('end_point_test_data',None)
batch_size = int(config['data']['batch_size'])
transformations = config['data'].get('transformations',None)
if transformations is not None:
    transform = FACTORY_DICT["transforms"][transformations["transform"]]
    transform_kwargs = transformations.get('info',{})
else:
    transform = None
    transform_kwargs = {}

#evaluation def
bQualitativa = bool(config['evaluation']['bQualitativa'])
bFID = bool(config['evaluation']['bFID'])
bSSIM = bool(config['evaluation']['bSSIM'])


##--------------------Preparing Objects for Evaluation--------------------

# Definição dos caminhos para localização do modelo e onde dados serão armazenados
trained_gen_dir = './' + model_gen_name + '/'
if use_best_version is True:
    trained_gen_path = trained_gen_dir+'models/'+ model_gen_name + '_gen_best.pt'
else:
    trained_gen_path = trained_gen_dir+'models/'+ model_gen_name + '_gen_trained.pt'

# Define gerador
gen = FACTORY_DICT["model_gen"]["Generator"]()
gen.load_state_dict(torch.load(trained_gen_path, weights_only=True))
gen.to(device)

# Obtenção dos dados de teste
dataset_test = FACTORY_DICT['dataset'][dataset_type](processed_data_folder=processed_data_folder,
                                                                mode='all_test',
                                                                start=start_point_test_data,
                                                                end=end_point_test_data,
                                                                transform=transform,
                                                                **transform_kwargs)

data_loader_test = DataLoader(dataset_test,
                            batch_size=batch_size,
                            shuffle=False)

###--------------------------Call for Evaluation--------------------------
evaluate(gen=gen, 
        trained_gen_dir=trained_gen_dir,
        device=device,
        dataset_test=dataset_test,
        data_loader_test=data_loader_test,
        batch_size=batch_size,
        bQualitativa=bQualitativa, 
        bFID=bFID, 
        bSSIM=bSSIM)





