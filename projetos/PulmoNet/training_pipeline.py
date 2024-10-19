import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
import random
random.seed(5)
import matplotlib.pyplot as plt
import numpy as np
import wandb
import os
import csv


from datasets import lungCTData
from model import Generator, Discriminator
from save_models_and_training import SaveBestModel, safe_save, SaveTrainingLosses, save_trained_models, delete_safe_save
from lr_scheduler_and_optim import LRScheduler,get_optimizer
from losses import get_criterion
from main import run_train_epoch, run_validation_epoch, valid_on_the_fly
from utils import clean_directory, read_yaml

config_path = input("Enter path for config.yaml: ")

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
config = read_yaml(file=config_path)

####----------------------Definition-----------------------------------
#names and directories
name_model = str(config['model']['name_model'])
dir_save_results = str(config['model'].get('dir_save_results', f'./{name_model}/'))
dir_save_models = dir_save_results+'models/'
dir_save_example = dir_save_results+'examples/'
new_model = bool(config['model'].get('new_model', True))

#data
processed_data_folder = str(config['data'].get('processed_data_folder','/mnt/shared/ctdata_thr25'))
start_point_train_data = int(config['data']['start_point_train_data'])
end_point_train_data = int(config['data']['end_point_train_data'])
start_point_validation_data = int(config['data']['start_point_validation_data'])
end_point_validation_data = int(config['data']['end_point_validation_data'])

#training
batch_size_train = int(config['training']['batch_size_train'])
batch_size_validation = int(config['training']['batch_size_validation'])
n_epochs = int(config['training']['n_epochs'])
steps_to_complete_bfr_upd_disc = int(config['training'].get('steps_to_complete_bfr_upd_disc',1))
steps_to_complete_bfr_upd_gen = int(config['training'].get('steps_to_complete_bfr_upd_gen',1))

#loss
criterion = get_criterion(config['loss']['criterion']['name'],**config['loss']['criterion'].get('info',{}))
if 'regularizer' in config['loss']:
    regularization_type = config['loss']['regularizer']['type']
    regularization_level = config['loss']['regularizer']['regularization']
else:
    regularization_type = None
    regularization_level = None

#models
gen = Generator().to(device)
disc = Discriminator().to(device)

#optimizer
optimizer_type = config['optimizer']['type']
initial_lr = config['optimizer']['lr']
gen_opt = get_optimizer(gen, optimizer_type, initial_lr, **config['optimizer'].get('info',{}))
disc_opt = get_optimizer(disc, optimizer_type, initial_lr, **config['optimizer'].get('info',{}))

#lr scheduler
use_lr_scheduler = bool(config['lr_scheduler']['activate'])
if use_lr_scheduler is True:
    scheduler_type = config['lr_scheduler']['scheduler_type']
    epoch_to_switch_to_lr_scheduler = config['lr_scheduler']['epoch_to_switch_to_lr_scheduler']
    gen_scheduler = LRScheduler(gen_opt,scheduler_type,**config['lr_scheduler'].get('info',{}))
    disc_scheduler =  LRScheduler(disc_opt,scheduler_type,**config['lr_scheduler'].get('info',{}))
else:
    gen_scheduler = None
    disc_scheduler = None
    epoch_to_switch_to_lr_scheduler = None

#saves
step_to_safe_save_models = int(config['save_models_and_results']['step_to_safe_save_models'])
save_training_losses = SaveTrainingLosses(dir_save_results=dir_save_results)
save_best_model = bool(config['save_models_and_results']['save_best_model'])
if save_best_model is True:
    best_model = SaveBestModel(dir_save_model=dir_save_models)

#wandb
use_wandb = bool(config['wandb']['activate'])


if use_wandb is True:
    wandb.init(
        # set the wandb project where this run will be logged
        project=name_model,
        # track hyperparameters and run metadata
        config={
            "datafolder": processed_data_folder,
            "idx_initial_train_data": start_point_train_data,
            "idx_final_train_data": end_point_train_data,
            "idx_initial_val_data": start_point_validation_data,
            "idx_final_val_data": end_point_validation_data,
            "batch_size_train": batch_size_train,
            "batch_size_val": batch_size_validation,
            "epochs": n_epochs,
            "steps_to_complete_bfr_upd_disc": steps_to_complete_bfr_upd_disc,
            "steps_to_complete_bfr_upd_gen": steps_to_complete_bfr_upd_gen,
            "criterion": config['loss']['criterion']['name'],
            "regularization_type": regularization_type,
            "regularization_level": regularization_level,
            "optimizer": optimizer_type,
            "initial_lr": initial_lr,
            "scheduler": gen_scheduler,
            "epoch_to_switch_to_lr_scheduler": epoch_to_switch_to_lr_scheduler
        }
    )

####----------------------Preparing objects-----------------------------------
dataset_train = lungCTData(processed_data_folder=processed_data_folder,
                           start=start_point_train_data,
                           end=end_point_train_data)
dataset_validation = lungCTData(processed_data_folder=processed_data_folder,
                                start=start_point_validation_data,
                                end=end_point_validation_data)

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

os.makedirs(dir_save_results, exist_ok=True)
if new_model is True:
    clean_directory(dir_save_results)
os.makedirs(dir_save_models, exist_ok=True)
os.makedirs(dir_save_example, exist_ok=True)
if new_model is True:
    with open(dir_save_results+'losses.csv', 'w', newline='') as csvfile:
        fieldnames = ['LossGenTrain', 'LossDiscTrain', 'LossGenVal', 'LossDiscVal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
else:
    print('Loading old model to keep training...')
    gen.load_state_dict(torch.load(str(config['model'].get('path_to_saved_model_gen',dir_save_models+f"{name_model}_gen_savesafe.pt")), weights_only=True))
    disc.load_state_dict(torch.load(str(config['model'].get('path_to_saved_model_disc',dir_save_models+f"{name_model}_disc_savesafe.pt")), weights_only=True))


####----------------------Training Loop-----------------------------------
for epoch in range(n_epochs):

    ####----------------------loops-----------------------------------
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
                                                      regularization_type=regularization_type,
                                                      regularization_level=regularization_level)
    mean_loss_train_gen_list.append(loss_train_gen)
    mean_loss_train_disc_list.append(loss_train_disc)

    loss_validation_gen, loss_validation_disc = run_validation_epoch(gen=gen,
                                                                     disc=disc,
                                                                     criterion=criterion,
                                                                     data_loader=data_loader_validation,
                                                                     epoch=epoch,
                                                                     device=device,
                                                                     use_wandb=use_wandb,
                                                                     regularization_type=regularization_type,
                                                                     regularization_level=regularization_level)
    mean_loss_validation_gen_list.append(loss_validation_gen)
    mean_loss_validation_disc_list.append(loss_validation_disc)

    valid_on_the_fly(gen=gen, disc=disc, data_loader=data_loader_validation, epoch=epoch, save_dir=dir_save_example,device=device)

    ###------------------------------------------savings----------------------------------
    if epoch % step_to_safe_save_models == 0:
        current_lr = gen_scheduler.get_last_lr()[0] if use_lr_scheduler else initial_lr
        safe_save(dir_save_models=dir_save_models, 
                    name_model=name_model,
                    gen=gen, 
                    disc=disc, 
                    epoch=epoch, 
                    current_lr=current_lr,
                    lr_scheduler=gen_scheduler)
    if (epoch % step_to_safe_save_models == 0) or (epoch == n_epochs-1):
        save_training_losses(mean_loss_train_gen_list=mean_loss_train_gen_list, 
                 mean_loss_train_disc_list=mean_loss_train_disc_list,
                 mean_loss_validation_gen_list=mean_loss_validation_gen_list,
                 mean_loss_validation_disc_list=mean_loss_validation_disc_list)
    
    if save_best_model is True:
        best_model(current_score=loss_validation_gen, 
                   name_model=name_model, 
                   gen=gen, 
                   disc=disc, 
                   epoch=epoch, 
                   use_wandb=use_wandb)

    ###------------------------------------------learning rate update----------------------------------
    if epoch >= epoch_to_switch_to_lr_scheduler:
        gen_scheduler.step()
        disc_scheduler.step()
        print("Current learning rate: gen: ", gen_scheduler.get_last_lr()[0], " disc: ", disc_scheduler.get_last_lr()[0])

####----------------------Finishing-----------------------------------
save_trained_models(dir_save_models=dir_save_models, name_model=name_model, gen=gen, disc=disc)
delete_safe_save(dir_save_models=dir_save_models, name_model=name_model)

if use_wandb is True:
    wandb.finish()

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(mean_loss_train_gen_list, label='Train')
ax[0].plot(mean_loss_validation_gen_list, label='Validation')
ax[1].plot(mean_loss_train_disc_list, label='Train')
ax[1].plot(mean_loss_validation_disc_list, label='Validation')
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
ax[0].set_title('Generator')
ax[1].set_title('Discriminator')
ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs')
plt.savefig(dir_save_results+'losses_evolution.png')