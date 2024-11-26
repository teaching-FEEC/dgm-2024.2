'''
@file segmentation_pipeline.py

@description
File for executing the training of the U-Net network for airway segmentation, 
with and without the weights of the best PulmoNet GAN generator. 
'''

import torch
from torch.utils.data import DataLoader
import os

from constants import *
from save_models_and_training import safe_save_unet, save_trained_models_unet, delete_safe_save_unet
from main_functions import run_train_epoch_unet, run_validation_epoch_unet, valid_on_the_fly_unet
from utils import read_yaml, plot_training_evolution_unet, retrieve_metrics_from_csv, prepare_environment_for_new_model, resume_training_unet, EarlyStopping

config_path = input("Enter path for YAML file with training description: ")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = read_yaml(file=config_path)

####----------------------Definition-----------------------------------
#names and directories
#dir_save_results will store all training info
name_model = str(config['model']['name_model'])
dir_save_results = str(config['model'].get('dir_save_results',
                                            f'./{name_model}/'))
dir_save_models = dir_save_results+'models/'
dir_save_example = dir_save_results+'examples/'
dir_save_test = dir_save_results+'test/'
#if model is not new, will resume training from a given model
new_model = bool(config['model'].get('new_model', True))
#if fine_tunning is True, will allocate the weights of a given model
fine_tunning = bool(config['model'].get('fine_tunning', True))
trained_gen_path = str(config['model']['path_to_saved_model'])
#if freeze_layers is True, we don't learn the parameters in the encoder part of the generator
freeze_layers = bool(config['model'].get('freeze_layers', True))

#models
unet = FACTORY_DICT["model_unet"]["Unet"](use_as_unet=True).to(device)
if fine_tunning is True:
    unet.load_state_dict(torch.load(trained_gen_path, weights_only=True, map_location=torch.device(device)))
    unet.to(device)
    unet.train() # Set the model to training mode

    if freeze_layers is True:
        # Freeze the encoder layers
        for param in unet.conv1.parameters():
            param.requires_grad = False
        for param in unet.conv2.parameters():
            param.requires_grad = False
        for param in unet.conv3.parameters():
            param.requires_grad = False
        for param in unet.conv4.parameters():
            param.requires_grad = False
        for param in unet.conv5.parameters():
            param.requires_grad = False
        for param in unet.conv6.parameters():
            param.requires_grad = False
        for param in unet.conv7.parameters():
            param.requires_grad = False
        for param in unet.conv8.parameters():
            param.requires_grad = False

#path to data: should be a directiory with folders: 'seg_train' and 'seg_val'
#inside of each folder should be 3 folders: 'images', 'labels' and 'lungs'
processed_data_folder = str(config['data']['processed_data_folder'])
print(processed_data_folder)
dataset_type = str(config['data']['dataset'])
print(dataset_type)
start_point_train_data = int(config['data']['start_point_train_data'])
print(start_point_train_data)
end_point_train_data = int(config['data']['end_point_train_data'])
start_point_validation_data = int(config['data']['start_point_validation_data'])
end_point_validation_data = int(config['data']['end_point_validation_data'])

#training hyperparameters
batch_size_train = int(config['training']['batch_size_train'])
batch_size_validation = int(config['training']['batch_size_validation'])
n_epochs = int(config['training']['n_epochs'])
#transformations to apply to images in dataset processing
transformations = config['training'].get('transformations',None)
if transformations is not None:
    transform = FACTORY_DICT["transforms"][transformations["transform"]]
    transform_kwargs = transformations.get('info',{})
else:
    transform = None
    transform_kwargs = {}

#if b_early_stopping allows for early stopping strategy
b_early_stopping = bool(config['training']['early_stopping'])
if b_early_stopping is True:
    patience = int(config['training']['patience'])
    delta = int(config['training']['delta'])
    #check class in utils.py
    early_stopping = EarlyStopping(patience=patience, delta=delta)

#loss
criterion = FACTORY_DICT["criterion"][config['loss']['criterion']['name']](**config['loss']['criterion'].get('info',{}))

#optimizer
optimizer_type = config['optimizer']['type']
initial_lr = config['optimizer']['lr']
unet_opt = FACTORY_DICT['optimizer'][optimizer_type](unet.parameters(),
                                                    lr=initial_lr,
                                                    **config['optimizer'].get('info',{}))

#saves
#save best model will consider the minimum of validation loss
#safe save: in case something goes wrong and the training must be interrupted, you don't loose everything
step_to_safe_save_models = int(config['save_models_and_results']['step_to_safe_save_models'])
save_training_losses = FACTORY_DICT["savelosses"]["SaveUnetTrainingLosses"](dir_save_results=dir_save_results)
save_best_model = bool(config['save_models_and_results']['save_best_model'])
if save_best_model is True:
    best_model = FACTORY_DICT["savebest"]["SaveBestUnetModel"](dir_save_model=dir_save_models)


####----------------------Preparing objects-----------------------------------

dataset_train = FACTORY_DICT["dataset"][dataset_type](
                            processed_data_folder=processed_data_folder,
                            mode="seg_train",
                            start=start_point_train_data,
                            end=end_point_train_data,
                            transform=transform,
                            **transform_kwargs)
dataset_validation = FACTORY_DICT["dataset"][dataset_type](
                            processed_data_folder=processed_data_folder,
                            mode="seg_val",
                            start=start_point_validation_data,
                            end=end_point_validation_data,
                            transform=transform,
                            **transform_kwargs)

data_loader_train = DataLoader(dataset_train,
                               batch_size=batch_size_train,
                               shuffle=True)
data_loader_validation = DataLoader(dataset_validation,
                                    batch_size=batch_size_validation,
                                    shuffle=True)


mean_loss_train_unet_list = []
mean_loss_validation_unet_list = []

#create folder or empties an existing folder if model is new
prepare_environment_for_new_model(new_model=new_model, 
                                  dir_save_results=dir_save_results,
                                  dir_save_models=dir_save_models,
                                  dir_save_example=dir_save_example)
if new_model is True:
    #creates CSV file to store losses
    save_training_losses.initialize_losses_file()
else:
    #reloads training objects
    epoch_resumed_from = resume_training_unet(dir_save_models=dir_save_models, 
                                                name_model=name_model, 
                                                unet=unet, 
                                                unet_opt=unet_opt, 
                                                config=config)
    

####----------------------Training Loop-----------------------------------
for epoch in range(n_epochs):

    ####----------------------loops-----------------------------------
    #update weights
    loss_train_unet = run_train_epoch_unet(unet=unet,
                                            criterion=criterion,
                                            data_loader=data_loader_train,
                                            unet_opt=unet_opt,
                                            epoch=epoch,
                                            device=device)
    mean_loss_train_unet_list.append(loss_train_unet)

    #check performance on valdiation data
    loss_validation_unet = run_validation_epoch_unet(unet=unet,
                                                    criterion=criterion,
                                                    data_loader=data_loader_validation,
                                                    epoch=epoch,
                                                    device=device)
    mean_loss_validation_unet_list.append(loss_validation_unet)

    #save an output sample
    if (new_model is True) or (epoch_resumed_from is None):
        epoch_to_appear_for_ref = epoch
    else:
        epoch_to_appear_for_ref = epoch+epoch_resumed_from
    valid_on_the_fly_unet(unet=unet, data_loader=data_loader_validation, epoch=epoch_to_appear_for_ref, save_dir=dir_save_example,device=device)

    ###------------------------------------------savings----------------------------------
    #stores model and optimizer
    if epoch % step_to_safe_save_models == 0:
        current_lr =  initial_lr
        safe_save_unet(dir_save_models=dir_save_models, 
                        name_model=name_model,
                        unet=unet, 
                        epoch=epoch, 
                        unet_optimizer=unet_opt,
                        current_lr=current_lr)
    #save losses in CSV file
    if (epoch % step_to_safe_save_models == 0) or (epoch == n_epochs-1):
        save_training_losses(mean_loss_train_unet_list=mean_loss_train_unet_list,
                             mean_loss_validation_unet_list=mean_loss_validation_unet_list)
        
    #checks if validation loss got smaller than the best value
    #if so, saves current model
    if save_best_model is True:
        best_model(current_score=loss_validation_unet, 
                   name_model=name_model, 
                   unet=unet, 
                   epoch=epoch, 
                   use_wandb=False)
        
    ###------------------------------------------early_stopping----------------------------------
    if b_early_stopping is True:
        early_stopping(loss_validation_unet, unet)
        if early_stopping.early_stop:
            print("Early stopping")
            delete_safe_save_unet(dir_save_models=dir_save_models, name_model=name_model)          
            break

####----------------------Finishing-----------------------------------
save_trained_models_unet(dir_save_models=dir_save_models, name_model=name_model, unet=unet)
delete_safe_save_unet(dir_save_models=dir_save_models, name_model=name_model)

if new_model is True:
    plot_training_evolution_unet(path=dir_save_results,
                            mean_loss_train_unet_list=mean_loss_train_unet_list,
                            mean_loss_validation_unet_list=mean_loss_validation_unet_list)
else:
    if os.path.isfile(dir_save_results+'losses.csv'):
        losses = retrieve_metrics_from_csv(path_file=dir_save_results+'losses.csv')
        plot_training_evolution_unet(path=dir_save_results,
                            mean_loss_train_unet_list=losses['LossUnetTrain'],
                            mean_loss_validation_unet_list=losses['LossUnetVal'])