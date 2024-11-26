#Evaluate PulmoNet for the generation of CT images (only)
#or evaluate U-Net

import torch
from torch.utils.data import DataLoader
import numpy as np
import os


from constants import *
from metrics import my_fid_pipeline, my_ssim_pipeline, dice_coefficient_score_calculation
from utils import read_yaml, plt_save_example_synth_during_test, save_quantitative_results, plt_save_example_airways_img, save_quantitative_results_unet


#-------------------------------------Function to evaluate PulmoNet-----------------------------------
def evaluate(gen, 
             trained_gen_dir,
             device,
             dataset_test,
             data_loader_test,
             batch_size = 10,
             bQualitativa=True, 
             bFID=True, 
             bSSIM=True):

    # directory where created images are stored
    dir_to_save_gen_imgs = trained_gen_dir+'generated_imgs/'
    # JSON with quantitative metrics
    path_to_save_metrics = trained_gen_dir+'quantitative_metrics.json'
    os.makedirs(dir_to_save_gen_imgs, exist_ok=True)

    # Set model to evaluate config
    gen.eval()

    # Generates images for qualitative evaluation (visual)
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
                                                           input_mask_ref=np.squeeze(input_mask.detach().cpu().numpy())[0, :, :],
                                                        gen_img_ref=np.squeeze(gen_img.detach().cpu().numpy())[0, :, :],
                                                        save_dir=dir_to_save_gen_imgs,
                                                        img_idx=counter)
                    else:
                        skip = skip - 1
                    if skip == 0:
                        skip=20

    # FID between real and fake images
    if (bFID is True):
        print('Calculating FID...')
        fid_value = my_fid_pipeline(dataset_test, data_loader_test, device, gen, batch_size)
    else:
        fid_value = np.nan

    # SSIM between real and fake images
    if (bSSIM is True):
        print('Calculating SSIM...')
        # Full image
        ssim_complete, luminance_complete, contraste_complete, struct_similarity_complete = my_ssim_pipeline(dataset_test, 
                                                                                                            data_loader_test, 
                                                                                                            device, gen, batch_size, 
                                                                                                            bComplete=True)
        # Only center (256x256)
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

    # Store results
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


#-------------------------------------Function to evaluate U-Net-----------------------------------
def evaluate_seg_net(model, data_loader_test, dir_save_test):
    # directory where created images are stored
    dir_to_save_gen_imgs = trained_gen_dir+'generated_imgs/'
    # JSON with quantitative metrics
    path_to_save_metrics = trained_gen_dir+'quantitative_metrics.json'
    os.makedirs(dir_to_save_gen_imgs, exist_ok=True)

    # Set model to evaluate config
    gen.eval()

    # Generates images for qualitative evaluation (visual)
    print('Generating examples for qualitative analysis...')
    counter = 0
    skip = 20
    with torch.no_grad():
        for batch in data_loader_test:
            if counter <= 20: 
                if skip == 20:

                    input_img_batch = batch[0]
                    input_airway_batch = batch[1]
                    
                    input_img = input_img_batch.to(device)
                    input_airway = input_airway_batch.to(device)
                    
                    gen_seg = model(input_img)
                    counter += 1

                    plt_save_example_airways_img(input_img_ref=np.squeeze(input_img.detach().cpu().numpy())[0, :, :],
                                                input_airway_ref=np.squeeze(input_airway.detach().cpu().numpy())[0, :, :],
                                                gen_seg_ref=np.squeeze(gen_seg.detach().cpu().numpy())[0, :, :],
                                                save_dir=dir_to_save_gen_imgs,
                                                img_idx=counter)
                else:
                    skip = skip - 1
                if skip == 0:
                    skip=20
        
        #DICE of segmented airways
        print('Calculating DICE...')
        dice =  dice_coefficient_score_calculation(pred=input_airway.detach().cpu().numpy(), label=input_img.detach().cpu().numpy())
        print(dice)

    # Store results
    save_quantitative_results_unet(dice = dice, save_path=path_to_save_metrics)
    print("Evaluation done!")


#-------------------------------------Read YAML file with configurations-----------------------------------
config_path = input("Enter path for YAML file with evaluation description: ")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = read_yaml(file=config_path)

##--------------------Definitions--------------------
#Generator model
model_gen_name = str(config['model']['name_model'])
trained_gen_dir = str(config['model'].get('dir_trained_model',
                                            f'./{model_gen_name}/'))
#if True use model marked as 'best' instead of 'trained' (last epoch)
use_best_version = bool(config['model']['use_best_version'])

#data for evaluation
#assumes a folder with a foder 'all_test' inside, and inside of this one, 
#three folders: images, lungs, labels
processed_data_folder = str(config['data'].get('processed_data_folder',
                                               '/mnt/shared/ctdata_thr25'))
#dataset as defined in constants.py and datasets.py
dataset_type = str(config['data'].get('dataset',
                                    'lungCTData'))
#if None will use all files available in folder
start_point_test_data = config['data'].get('start_point_test_data',None)
end_point_test_data = config['data'].get('end_point_test_data',None)
batch_size = int(config['data']['batch_size'])
transformations = config['data'].get('transformations',None)
#transformations to apply to images in dataset processing
if transformations is not None:
    transform = FACTORY_DICT["transforms"][transformations["transform"]]
    transform_kwargs = transformations.get('info',{})
else:
    transform = None
    transform_kwargs = {}

#Definition of the metrics evaluated in this pipeline
bQualitativa = bool(config['evaluation']['bQualitativa'])
bFID = bool(config['evaluation']['bFID'])
bSSIM = bool(config['evaluation']['bSSIM'])

bDice = bool(config['evaluation']['bDice']) # only for the segmentation net


##--------------------Preparing Objects for Evaluation--------------------
if bDice is False:
    if use_best_version is True:
        trained_gen_path = trained_gen_dir+'models/'+ model_gen_name + '_gen_best.pt'
    else:
        trained_gen_path = trained_gen_dir+'models/'+ model_gen_name + '_gen_trained.pt'
else:
    if use_best_version is True:
        trained_gen_path = trained_gen_dir+'models/'+ model_gen_name + '_unet_best.pt'
    else:
        trained_gen_path = trained_gen_dir+'models/'+ model_gen_name + '_unet_trained.pt'

# generator
gen = FACTORY_DICT["model_gen"]["Generator"]()
gen.load_state_dict(torch.load(trained_gen_path, weights_only=True, map_location=torch.device(device)))
gen.to(device)

#get data for test
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
# If its not the utility test, calculate the qualitative and quantitative (FID and SSIM) metrics
if bDice is False: 
    evaluate(gen=gen, 
            trained_gen_dir=trained_gen_dir,
            device=device,
            dataset_test=dataset_test,
            data_loader_test=data_loader_test,
            batch_size=batch_size,
            bQualitativa=bQualitativa, 
            bFID=bFID, 
            bSSIM=bSSIM)
# Otherwise, calculate the Dice for the segmentation net
else:
    evaluate_seg_net(model=gen, data_loader_test=data_loader_test, dir_save_test=trained_gen_dir)