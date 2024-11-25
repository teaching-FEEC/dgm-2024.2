#Pipeline to train PulmoNet
#To generate only CT images or to generate both CT images and airway segmentation
#Must receive a YAML file with training definition

import torch
from torch.utils.data import DataLoader
import wandb
import os

from constants import *
from save_models_and_training import safe_save, save_trained_models, delete_safe_save
from lr_scheduler import LRScheduler
from losses import Regularizer
from main_functions import run_train_epoch, run_validation_epoch, valid_on_the_fly
from utils import read_yaml, plot_training_evolution, retrieve_metrics_from_csv, prepare_environment_for_new_model, resume_training, start_from_pretrained_model

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
#if model is not new, will resume training from a given model
new_model = bool(config['model'].get('new_model', True))
#if use_pretrained_model is True, will allocate the weights of a given model
use_pretrained_model= bool(config['model'].get('use_pretrained_model', False))
#generate only CT images or CT images and airway segmentatation
generate_airway_segmentation = bool(config['model'].get('generate_airway_segmentation', False))

#models
gen = FACTORY_DICT["model_gen"]["Generator"](generate_airway_segmentation=generate_airway_segmentation).to(device)
disc = FACTORY_DICT["model_disc"]["Discriminator"](generate_airway_segmentation=generate_airway_segmentation).to(device)

#path to data: should be a directiory with folders: 'gan_train' and 'gan_val'
#inside of each folder should be 3 folders: 'images', 'labels' and 'lungs'
processed_data_folder = str(config['data'].get('processed_data_folder',
                                               '/mnt/shared/divided_data'))
#dataset as defined in constants.py and datasets.py
dataset_type = str(config['data'].get('dataset',
                                    'lungCTData'))
#if None will use all files available in folders
start_point_train_data = config['data'].get('start_point_train_data',None)
end_point_train_data = config['data'].get('end_point_train_data',None)
start_point_validation_data = config['data'].get('start_point_validation_data',None)
end_point_validation_data = config['data'].get('end_point_validation_data',None)

#training hyperparameters
batch_size_train = int(config['training']['batch_size_train'])
batch_size_validation = int(config['training']['batch_size_validation'])
n_epochs = int(config['training']['n_epochs'])
#steps_to_complete_bfr_upd_disc: the amount of steps the GENERATOR takes BEFORE the disc gets updated
#steps_to_complete_bfr_upd_gen: the amount of steps the DISCRIMINATOR takes BEFORE the gen gets updated
steps_to_complete_bfr_upd_disc = int(config['training'].get('steps_to_complete_bfr_upd_disc',1))
steps_to_complete_bfr_upd_gen = int(config['training'].get('steps_to_complete_bfr_upd_gen',1))
#transformations to apply to images in dataset processing
transformations = config['training'].get('transformations',None)
if transformations is not None:
    transform = FACTORY_DICT["transforms"][transformations["transform"]]
    transform_kwargs = transformations.get('info',{})
else:
    transform = None
    transform_kwargs = {}

#loss
criterion = FACTORY_DICT["criterion"][config['loss']['criterion']['name']](**config['loss']['criterion'].get('info',{}))
if 'regularizer' in config['loss']:
    regularization_type = config['loss']['regularizer']['type']
    regularizer = Regularizer(regularization_type)
    regularization_level = config['loss']['regularizer']['regularization']
else:
    regularization_type = None
    regularizer = None
    regularization_level = None

#optimizer
optimizer_type = config['optimizer']['type']
initial_lr = config['optimizer']['lr']
gen_opt = FACTORY_DICT['optimizer'][optimizer_type](gen.parameters(),
                                                    lr=initial_lr,
                                                    **config['optimizer'].get('info',{}))
disc_opt = FACTORY_DICT['optimizer'][optimizer_type](disc.parameters(),
                                                    lr=initial_lr,
                                                    **config['optimizer'].get('info',{}))

#lr scheduler
use_lr_scheduler = bool(config['lr_scheduler']['activate'])
if use_lr_scheduler is True:
    scheduler_type = config['lr_scheduler']['scheduler_type']
    epoch_to_switch_to_lr_scheduler = config['lr_scheduler']['epoch_to_switch_to_lr_scheduler']
    gen_scheduler = LRScheduler(gen_opt,scheduler_type,**config['lr_scheduler'].get('info',{}))
    disc_scheduler =  LRScheduler(disc_opt,scheduler_type,**config['lr_scheduler'].get('info',{}))
else:
    scheduler_type = None
    gen_scheduler = None
    disc_scheduler = None
    epoch_to_switch_to_lr_scheduler = None

#saves
#save best model will consider the minimum of validation loss
#safe save: in case something goes wrong and the training must be interrupted, you don't loose everything
step_to_safe_save_models = int(config['save_models_and_results']['step_to_safe_save_models'])
save_training_losses = FACTORY_DICT["savelosses"]["SaveTrainingLosses"](dir_save_results=dir_save_results)
save_best_model = bool(config['save_models_and_results']['save_best_model'])
if save_best_model is True:
    best_model = FACTORY_DICT["savebest"]["SaveBestModel"](dir_save_model=dir_save_models)

#wandb
use_wandb = bool(config['wandb']['activate'])

if use_wandb is True:
    wandb.init(
        # set the wandb project where this run will be logged
        project=name_model,
        # track hyperparameters and run metadata
        config={
            "datafolder": processed_data_folder,
            "dataset": dataset_type,
            "idx_initial_train_data": start_point_train_data,
            "idx_final_train_data": end_point_train_data,
            "idx_initial_val_data": start_point_validation_data,
            "idx_final_val_data": end_point_validation_data,
            "batch_size_train": batch_size_train,
            "batch_size_val": batch_size_validation,
            "epochs": n_epochs,
            "steps_to_complete_bfr_upd_disc": steps_to_complete_bfr_upd_disc,
            "steps_to_complete_bfr_upd_gen": steps_to_complete_bfr_upd_gen,
            "transform": transform,
            "criterion": config['loss']['criterion']['name'],
            "regularization_type": regularization_type,
            "regularization_level": regularization_level,
            "optimizer": optimizer_type,
            "initial_lr": initial_lr,
            "scheduler": scheduler_type,
            "epoch_to_switch_to_lr_scheduler": epoch_to_switch_to_lr_scheduler,
            "model_generate_airway": generate_airway_segmentation
        }
    )


####----------------------Preparing objects-----------------------------------

dataset_train = FACTORY_DICT["dataset"][dataset_type](
                            processed_data_folder=processed_data_folder,
                            mode='gan_train',
                            start=start_point_train_data,
                            end=end_point_train_data,
                            transform=transform,
                            **transform_kwargs)
dataset_validation = FACTORY_DICT["dataset"][dataset_type](
                            processed_data_folder=processed_data_folder,
                            mode='gan_val',
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


mean_loss_train_gen_list = []
mean_loss_validation_gen_list = []
mean_loss_train_disc_list = []
mean_loss_validation_disc_list = []

#create folder or empties an existing folder if model is new
prepare_environment_for_new_model(new_model=new_model, 
                                  dir_save_results=dir_save_results,
                                  dir_save_models=dir_save_models,
                                  dir_save_example=dir_save_example)
if new_model is True:
    #creates CSV file to store losses
    save_training_losses.initialize_losses_file()
    if use_pretrained_model is True:
        #loads network weights to fine-tune
        start_from_pretrained_model(gen=gen, disc=disc, config=config)
else:
    #reloads training objects
    epoch_resumed_from = resume_training(dir_save_models=dir_save_models, 
                                        name_model=name_model, 
                                        gen=gen, 
                                        disc=disc, 
                                        gen_opt=gen_opt, 
                                        disc_opt=disc_opt, 
                                        config=config, 
                                        use_lr_scheduler=use_lr_scheduler, 
                                        gen_scheduler=gen_scheduler, 
                                        disc_scheduler=disc_scheduler)
    

####----------------------Training Loop-----------------------------------
for epoch in range(n_epochs):

    ####----------------------loops-----------------------------------
    #update weights
    loss_train_gen, loss_train_disc = run_train_epoch(gen=gen,
                                                      disc=disc,
                                                      criterion=criterion,
                                                      data_loader=data_loader_train,
                                                      disc_opt=disc_opt,
                                                      gen_opt=gen_opt,
                                                      epoch=epoch,
                                                      steps_to_complete_bfr_upd_disc=steps_to_complete_bfr_upd_disc,
                                                      steps_to_complete_bfr_upd_gen=steps_to_complete_bfr_upd_gen,
                                                      device=device,
                                                      use_wandb=use_wandb,
                                                      regularizer=regularizer,
                                                      regularization_level=regularization_level,
                                                      generate_airway_segmentation=generate_airway_segmentation)
    mean_loss_train_gen_list.append(loss_train_gen)
    mean_loss_train_disc_list.append(loss_train_disc)

    #check performance on valdiation data
    loss_validation_gen, loss_validation_disc = run_validation_epoch(gen=gen,
                                                                     disc=disc,
                                                                     criterion=criterion,
                                                                     data_loader=data_loader_validation,
                                                                     epoch=epoch,
                                                                     device=device,
                                                                     use_wandb=use_wandb,
                                                                     regularizer=regularizer,
                                                                     regularization_level=regularization_level,
                                                                     generate_airway_segmentation=generate_airway_segmentation)
    mean_loss_validation_gen_list.append(loss_validation_gen)
    mean_loss_validation_disc_list.append(loss_validation_disc)

    #save a created sample
    if (new_model is True) or (epoch_resumed_from is None):
        epoch_to_appear_for_ref = epoch
    else:
        epoch_to_appear_for_ref = epoch+epoch_resumed_from
    valid_on_the_fly(gen=gen, 
                     disc=disc, 
                     data_loader=data_loader_validation, 
                     epoch=epoch_to_appear_for_ref, 
                     save_dir=dir_save_example,
                     device=device, 
                     generate_airway_segmentation=generate_airway_segmentation)

    ###------------------------------------------savings----------------------------------
    #stores models, optimizers and lr schedulers
    if epoch % step_to_safe_save_models == 0:
        current_lr = gen_scheduler.get_last_lr()[0] if use_lr_scheduler else initial_lr
        safe_save(dir_save_models=dir_save_models, 
                    name_model=name_model,
                    gen=gen, 
                    disc=disc, 
                    epoch=epoch, 
                    gen_optimizer=gen_opt,
                    disc_optimizer=disc_opt,
                    current_lr=current_lr,
                    gen_scheduler=gen_scheduler,
                    disc_scheduler=disc_scheduler)
    #save losses in CSV file
    if (epoch % step_to_safe_save_models == 0) or (epoch == n_epochs-1):
        save_training_losses(mean_loss_train_gen_list=mean_loss_train_gen_list, 
                 mean_loss_train_disc_list=mean_loss_train_disc_list,
                 mean_loss_validation_gen_list=mean_loss_validation_gen_list,
                 mean_loss_validation_disc_list=mean_loss_validation_disc_list)
    #checks if validation loss of generator got smaller than the best value
    #if so, saves current model
    if save_best_model is True:
        best_model(current_score=loss_validation_gen, 
                   name_model=name_model, 
                   gen=gen, 
                   disc=disc, 
                   epoch=epoch, 
                   use_wandb=use_wandb)

    ###------------------------------------------learning rate update----------------------------------
    if (use_lr_scheduler is True) and (epoch >= epoch_to_switch_to_lr_scheduler):
        gen_scheduler.step()
        disc_scheduler.step()
        print("Current learning rate: gen: ", gen_scheduler.get_last_lr()[0], " disc: ", disc_scheduler.get_last_lr()[0])


####----------------------Finishing-----------------------------------
save_trained_models(dir_save_models=dir_save_models, name_model=name_model, gen=gen, disc=disc)
#delete safe saves to save space
delete_safe_save(dir_save_models=dir_save_models, name_model=name_model)

if use_wandb is True:
    wandb.finish()

if new_model is True:
    plot_training_evolution(path=dir_save_results,
                            mean_loss_train_gen_list=mean_loss_train_gen_list,
                            mean_loss_validation_gen_list=mean_loss_validation_gen_list,
                            mean_loss_train_disc_list=mean_loss_train_disc_list,
                            mean_loss_validation_disc_list=mean_loss_validation_disc_list)
else:
    if os.path.isfile(dir_save_results+'losses.csv'):
        losses = retrieve_metrics_from_csv(path_file=dir_save_results+'losses.csv')
        plot_training_evolution(path=dir_save_results,
                            mean_loss_train_gen_list=losses['LossGenTrain'],
                            mean_loss_validation_gen_list=losses['LossGenVal'],
                            mean_loss_train_disc_list=losses['LossDiscTrain'],
                            mean_loss_validation_disc_list=losses['LossDiscVal'])