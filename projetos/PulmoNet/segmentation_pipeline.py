'''
@file segmentation_pipeline.py

@description
Arquivo para execução do treinamento da rede U-Net para segmentação das vias aéreas,
com e sem os pesos do melhor gerador da GAN da PulmoNet.
O desempenho desta rede U-Net sem pesos será comparado com o fine-tunning do gerador
da PulmoNet para a tarefa de segmentação das vias aéreas.
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
name_model = str(config['model']['name_model'])
dir_save_results = str(config['model'].get('dir_save_results',
                                            f'./{name_model}/'))
dir_save_models = dir_save_results+'models/'
dir_save_example = dir_save_results+'examples/'
dir_save_test = dir_save_results+'test/'
new_model = bool(config['model'].get('new_model', True))

fine_tunning = bool(config['model'].get('fine_tunning', True))
trained_gen_path = str(config['model']['trained_gen_path'])
freeze_layers = bool(config['model'].get('freeze_layers', True))

#models
unet = FACTORY_DICT["model_unet"]["Unet"]().to(device)
if fine_tunning is True:
    unet.load_state_dict(torch.load(trained_gen_path, weights_only=False, map_location=torch.device(device)))
    unet.to(device)
    unet.eval()

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

#data
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
transformations = config['training'].get('transformations',None)
if transformations is not None:
    transform = FACTORY_DICT["transforms"][transformations["transform"]]
    transform_kwargs = transformations.get('info',{})
else:
    transform = None
    transform_kwargs = {}

b_early_stopping = bool(config['training']['early_stopping'])
if b_early_stopping is True:
    patience = int(config['training']['patience'])
    delta = int(config['training']['delta'])
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

prepare_environment_for_new_model(new_model=new_model, 
                                  dir_save_results=dir_save_results,
                                  dir_save_models=dir_save_models,
                                  dir_save_example=dir_save_example)
if new_model is True:
    save_training_losses.initialize_losses_file()
else:
    epoch_resumed_from = resume_training_unet(dir_save_models=dir_save_models, 
                                                name_model=name_model, 
                                                unet=unet, 
                                                unet_opt=unet_opt, 
                                                config=config)
    

####----------------------Training Loop-----------------------------------
for epoch in range(n_epochs):

    ####----------------------loops-----------------------------------
    loss_train_unet = run_train_epoch_unet(unet=unet,
                                            criterion=criterion,
                                            data_loader=data_loader_train,
                                            unet_opt=unet_opt,
                                            epoch=epoch,
                                            device=device)
    mean_loss_train_unet_list.append(loss_train_unet)

    loss_validation_unet = run_validation_epoch_unet(unet=unet,
                                                    criterion=criterion,
                                                    data_loader=data_loader_validation,
                                                    epoch=epoch,
                                                    device=device)
    mean_loss_validation_unet_list.append(loss_validation_unet)

    if (new_model is True) or (epoch_resumed_from is None):
        epoch_to_appear_for_ref = epoch
    else:
        epoch_to_appear_for_ref = epoch+epoch_resumed_from
    valid_on_the_fly_unet(unet=unet, data_loader=data_loader_validation, epoch=epoch_to_appear_for_ref, save_dir=dir_save_example,device=device)

    ###------------------------------------------savings----------------------------------
    if epoch % step_to_safe_save_models == 0:
        current_lr =  initial_lr
        safe_save_unet(dir_save_models=dir_save_models, 
                        name_model=name_model,
                        unet=unet, 
                        epoch=epoch, 
                        unet_optimizer=unet_opt,
                        current_lr=current_lr)
    if (epoch % step_to_safe_save_models == 0) or (epoch == n_epochs-1):
        save_training_losses(mean_loss_train_unet_list=mean_loss_train_unet_list,
                             mean_loss_validation_unet_list=mean_loss_validation_unet_list)
    
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
            break

####----------------------Finishing-----------------------------------
save_trained_models_unet(dir_save_models=dir_save_models, name_model=name_model, unet=unet)
#delete_safe_save_unet(dir_save_models=dir_save_models, name_model=name_model)

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