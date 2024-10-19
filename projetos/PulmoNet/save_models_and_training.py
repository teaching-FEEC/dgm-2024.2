import torch
import json
import csv
import wandb
import os

class SaveTrainingLosses:
    def __init__(self, dir_save_results, save_count_idx=0):
        self.save_count_idx = save_count_idx
        self.dir_save_results = dir_save_results
    def __call__(self, mean_loss_train_gen_list, 
                 mean_loss_train_disc_list,
                 mean_loss_validation_gen_list,
                 mean_loss_validation_disc_list):
        with open(self.dir_save_results+'losses.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for i in range(self.save_count_idx, len(mean_loss_train_gen_list)):
                writer.writerow([mean_loss_train_gen_list[i],
                                 mean_loss_train_disc_list[i],
                                 mean_loss_validation_gen_list[i],
                                 mean_loss_validation_disc_list[i]])
        self.save_count_idx = len(mean_loss_train_gen_list)


def safe_save(dir_save_models, name_model,
              gen, disc, epoch, current_lr,
              lr_scheduler=None):
    torch.save(gen.state_dict(),
               f"{dir_save_models}{name_model}_gen_savesafe.pt")
    torch.save(disc.state_dict(),
               f"{dir_save_models}{name_model}_disc_savesafe.pt")
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(),
                f"{dir_save_models}{name_model}_scheduler_state_savesafe.pt")
    training_state = {
        'epoch': epoch,
        'learning_rate': current_lr
    }
    with open(f"{dir_save_models}{name_model}_training_state_savesafe", "w") as outfile:
        outfile.write(json.dumps(training_state, indent=4))

def delete_safe_save(dir_save_models, name_model):
    if os.path.isfile(f"{dir_save_models}{name_model}_gen_trained.pt") and os.path.isfile(f"{dir_save_models}{name_model}_disc_trained.pt"):
        if os.path.isfile(f"{dir_save_models}{name_model}_gen_savesafe.pt"):
            os.remove(f"{dir_save_models}{name_model}_gen_savesafe.pt")
        if os.path.isfile(f"{dir_save_models}{name_model}_disc_savesafe.pt"):
            os.remove(f"{dir_save_models}{name_model}_disc_savesafe.pt")
        if os.path.isfile(f"{dir_save_models}{name_model}_scheduler_state_savesafe.pt"):
            os.remove(f"{dir_save_models}{name_model}_scheduler_state_savesafe.pt")
        if os.path.isfile(f"{dir_save_models}{name_model}_training_state_savesafe"):
            os.remove(f"{dir_save_models}{name_model}_training_state_savesafe")

def save_trained_models(dir_save_models, name_model, gen, disc):
    torch.save(gen.state_dict(), f"{dir_save_models}{name_model}_gen_trained.pt")
    torch.save(disc.state_dict(), f"{dir_save_models}{name_model}_disc_trained.pt")


class SaveBestModel:
    def __init__(self, dir_save_model, best_score=float("inf")):
        self.best_score = best_score
        self.dir_save_model = dir_save_model

    def __call__(self, current_score, name_model, gen, disc, 
                 epoch, use_wandb=None):
        if current_score < self.best_score:
            self.best_score = current_score
            torch.save(gen.state_dict(), f"{self.dir_save_model}{name_model}_gen_best.pt")
            torch.save(disc.state_dict(), f"{self.dir_save_model}{name_model}_disc_best.pt")
            training_state = {
                'epoch': epoch,
                'best_score': self.best_score
            }
            with open(f"{self.dir_save_model}{name_model}_best_info", "w") as outfile:
                outfile.write(json.dumps(training_state, indent=4))
            if use_wandb is True:
                torch.save(gen.state_dict(), os.path.join(wandb.run.dir, "gen_best.pt"))
                torch.save(disc.state_dict(),  os.path.join(wandb.run.dir, "disc_best.pt"))
