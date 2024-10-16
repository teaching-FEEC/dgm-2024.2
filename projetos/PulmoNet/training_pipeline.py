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
from main import run_train_epoch, run_validation_epoch, valid_on_the_fly
from utils import clean_directory


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#dados
start_point_train_data = 0
end_point_train_data = 10000
start_point_validation_data = 10000
end_point_validation_data = 10500
#bacthes
batch_size_train = 20
batch_size_validation = 8
#learning param
n_epochs = 50
initial_lr = 0.0002
epoch_to_switch_to_lr_scheduler = 100
#loss
criterion = torch.nn.BCELoss()
regularization = 5
steps_to_complete_bfr_upd_disc = 1
steps_to_complete_bfr_upd_gen = 1
#safe save
step_to_safe_save_models = 10
#save results directory
new_model = True
dir_save_results = './model_reg_5_thr_25k/'
dir_save_models = dir_save_results+'models/'
dir_save_example = dir_save_results+'examples/'
name_model = 'model_reg_5_thr_25k'
processed_data_folder = '/mnt/shared/ctdata_thr25'
#connect to wandb
use_wandb = True

if use_wandb == True:
    wandb.init(
        # set the wandb project where this run will be logged
        project= name_model,
        # track hyperparameters and run metadata
        config={
        "datafolder": processed_data_folder[12:],
        "idx_initial_train_data":start_point_train_data,
        "idx_final_train_data": end_point_train_data,
        "idx_initial_val_data": start_point_validation_data,
        "idx_final_val_data": end_point_validation_data,
        "batch_size_train": batch_size_train,
        "batch_size_val": batch_size_validation,
        "epochs": n_epochs,
        "initial_lr": initial_lr,
        "epoch_to_switch_to_lr_scheduler": epoch_to_switch_to_lr_scheduler,
        "criterion": "BCELoss",
        "regularization":regularization,
        "steps_to_complete_bfr_upd_disc":steps_to_complete_bfr_upd_disc,
        "steps_to_complete_bfr_upd_gen":steps_to_complete_bfr_upd_gen
        }
    )

dataset_train = lungCTData(processed_data_folder=processed_data_folder,mode='train',start=start_point_train_data,end=end_point_train_data)
dataset_validation = lungCTData(processed_data_folder=processed_data_folder,mode='train',start=start_point_validation_data,end=end_point_validation_data)

data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
data_loader_validation = DataLoader(dataset_validation, batch_size=batch_size_validation, shuffle=True)

gen = Generator().to(device)
disc = Discriminator().to(device)

gen_opt = torch.optim.Adam(gen.parameters(), lr=initial_lr, betas=(0.5, 0.999))
disc_opt = torch.optim.Adam(disc.parameters(), lr=initial_lr, betas=(0.5, 0.999))
gen_scheduler = torch.optim.lr_scheduler.LinearLR(gen_opt, start_factor=1.0, end_factor=0.0, total_iters=50)
disc_scheduler = torch.optim.lr_scheduler.LinearLR(disc_opt, start_factor=1.0, end_factor=0.0, total_iters=50)

mean_loss_train_gen_list = []
mean_loss_validation_gen_list = []
mean_loss_train_disc_list = []
mean_loss_validation_disc_list = []
save_count_idx = 0

os.makedirs(dir_save_results, exist_ok=True)
if new_model == True:
    clean_directory(dir_save_results)
os.makedirs(dir_save_models, exist_ok=True)
os.makedirs(dir_save_example, exist_ok=True)
if new_model == True:
    with open(dir_save_results+'losses.csv', 'w', newline='') as csvfile:
        fieldnames = ['LossGenTrain', 'LossDiscTrain', 'LossGenVal', 'LoddDiscVal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

for epoch in range(n_epochs):

    loss_train_gen, loss_train_disc = run_train_epoch(gen=gen, disc=disc, criterion=criterion, regularization=regularization, 
                                        data_loader=data_loader_train, disc_opt=disc_opt, gen_opt=gen_opt, 
                                        epoch=epoch, steps_to_complete_bfr_upd_disc=steps_to_complete_bfr_upd_disc, 
                                        steps_to_complete_bfr_upd_gen=steps_to_complete_bfr_upd_gen, device=device,use_wandb=use_wandb)

    mean_loss_train_gen_list.append(loss_train_gen)
    mean_loss_train_disc_list.append(loss_train_disc)

    loss_validation_gen, loss_validation_disc = run_validation_epoch(gen=gen, disc=disc, criterion=criterion, regularization=regularization, 
                                                data_loader=data_loader_validation, epoch=epoch, device=device, use_wandb=use_wandb)

    mean_loss_validation_gen_list.append(loss_validation_gen)
    mean_loss_validation_disc_list.append(loss_validation_disc)

    valid_on_the_fly(gen=gen, disc=disc, data_loader=data_loader_validation, epoch=epoch,save_dir=dir_save_example,device=device)

    if epoch%step_to_safe_save_models == 0:
        torch.save(gen.state_dict(), f"{dir_save_models}{name_model}_gen_last_lr_{gen_scheduler.get_last_lr()[0]}_savesafe.pt")
        torch.save(disc.state_dict(), f"{dir_save_models}{name_model}__disc_last_lr_{disc_scheduler.get_last_lr()[0]}_savesafe.pt")
        with open(dir_save_results+'losses.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for i in range(save_count_idx,epoch+1):
                writer.writerow([mean_loss_train_gen_list[i], mean_loss_train_disc_list[i], 
                                mean_loss_validation_gen_list[i],mean_loss_validation_disc_list[i]])
        save_count_idx = epoch+1

    if epoch >= epoch_to_switch_to_lr_scheduler:
        gen_scheduler.step()
        disc_scheduler.setp()
        print("Current learning rate: gen: ", gen_scheduler.get_last_lr()[0], " disc: ", disc_scheduler.get_last_lr()[0])


torch.save(gen.state_dict(), f"{dir_save_models}{name_model}_gen_trained.pt")
torch.save(disc.state_dict(), f"{dir_save_models}{name_model}_disc_trained.pt")
if save_count_idx < n_epochs:
    with open(dir_save_results+'losses.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(save_count_idx,n_epochs):
            writer.writerow([mean_loss_train_gen_list[i], mean_loss_train_disc_list[i], 
                            mean_loss_validation_gen_list[i],mean_loss_validation_disc_list[i]])

if use_wandb == True:
    wandb.finish()

fig,ax = plt.subplots(1,2,figsize=(14,4))
ax[0].plot(mean_loss_train_gen_list, label= 'Train')
ax[0].plot(mean_loss_validation_gen_list, label='Validation')
ax[1].plot(mean_loss_train_disc_list,label='Train')
ax[1].plot(mean_loss_validation_disc_list,label='Validation')
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
ax[0].set_title('Generator')
ax[1].set_title('Discriminator')
ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs')
plt.savefig(dir_save_results+'_losses_evolution.png')