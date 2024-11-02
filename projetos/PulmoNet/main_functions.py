    
from tqdm import trange
import torch
import gc
from losses import get_gen_loss, get_disc_loss
from utils import plt_save_example_synth_img, set_requires_grad
import wandb

def run_train_epoch(gen, 
                    disc, 
                    criterion, 
                    data_loader, 
                    disc_opt, 
                    gen_opt, 
                    epoch, 
                    steps_to_complete_bfr_upd_disc, 
                    steps_to_complete_bfr_upd_gen, device,
                    use_wandb, 
                    regularizer=None,
                    regularization_level=None):

    mean_loss_gen = 0
    mean_loss_disc = 0

    counter_batches_used_to_upd_disc = 0
    counter_batches_used_to_upd_gen = 0

    counter_steps_before_upd_disc = 0
    counter_steps_before_upd_gen = 0
    

    with trange(len(data_loader), desc='Train Loop') as progress_bar:
        for batch_idx, batch in zip(progress_bar, data_loader):

            input_img_batch = batch[0]
            input_mask_batch = batch[1]

            input_img = input_img_batch.to(device)
            input_mask = input_mask_batch.to(device)
            
            if counter_steps_before_upd_disc == 0:
                set_requires_grad(model=disc,set_require_grad=True)
                disc_opt.zero_grad()
                disc_loss = get_disc_loss(gen=gen,disc=disc,criterion=criterion,
                                          input_mask=input_mask,input_img=input_img)
                disc_loss.backward(retain_graph=True)
                disc_opt.step()
                mean_loss_disc = mean_loss_disc + disc_loss.item() 
                counter_batches_used_to_upd_disc = counter_batches_used_to_upd_disc + 1
                counter_steps_before_upd_gen = counter_steps_before_upd_gen + 1
                if counter_steps_before_upd_gen == steps_to_complete_bfr_upd_gen:
                    counter_steps_before_upd_gen = 0
            
            if counter_steps_before_upd_gen == 0:
                set_requires_grad(model=disc,set_require_grad=False)
                gen_opt.zero_grad()
                gen_loss = get_gen_loss(gen=gen,disc=disc,criterion=criterion,input_mask=input_mask,
                                        input_img=input_img,regularizer=regularizer,
                                        regularization_level=regularization_level,device=device)
                gen_loss.backward(retain_graph=True)
                gen_opt.step()
                mean_loss_gen = mean_loss_gen + gen_loss.item() 
                counter_batches_used_to_upd_gen = counter_batches_used_to_upd_gen + 1
                counter_steps_before_upd_disc = counter_steps_before_upd_disc + 1
                if counter_steps_before_upd_disc == steps_to_complete_bfr_upd_disc:
                    counter_steps_before_upd_disc = 0

            progress_bar.set_postfix(
                desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(data_loader):d},'
                      f'gen loss: {mean_loss_gen / (counter_batches_used_to_upd_gen)},'
                      f'disc loss: {mean_loss_disc / (counter_batches_used_to_upd_disc)}'))
            
    if use_wandb == True:
        wandb.log({"gen_loss_train": mean_loss_gen/(counter_batches_used_to_upd_gen), 
                    "disc_loss_train": mean_loss_disc/(counter_batches_used_to_upd_disc)})
    
    return (mean_loss_gen/(counter_batches_used_to_upd_gen)), (mean_loss_disc/(counter_batches_used_to_upd_disc))

def run_validation_epoch(gen, 
                         disc, 
                         criterion, 
                         data_loader, 
                         epoch, 
                         device, 
                         use_wandb, 
                         regularizer=None,
                         regularization_level=None):

    mean_loss_gen = 0
    mean_loss_disc = 0

    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
        with trange(len(data_loader), desc='Validation Loop') as progress_bar:
            for batch_idx, batch in zip(progress_bar, data_loader):

                input_img_batch = batch[0]
                input_mask_batch = batch[1]

                input_img = input_img_batch.to(device)
                input_mask = input_mask_batch.to(device)
                disc_loss = get_disc_loss(gen=gen,disc=disc,criterion=criterion,
                                          input_mask=input_mask,input_img=input_img)
                mean_loss_disc = mean_loss_disc + disc_loss.item() 
                gen_loss = get_gen_loss(gen=gen,disc=disc,criterion=criterion,input_mask=input_mask,
                                        input_img=input_img,regularizer=regularizer,
                                        regularization_level=regularization_level,device=device)
                mean_loss_gen = mean_loss_gen + gen_loss.item() 

                progress_bar.set_postfix(
                desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(data_loader):d},'
                      f'gen loss: {mean_loss_gen / (batch_idx + 1)},'
                      f'disc loss: {mean_loss_disc / (batch_idx + 1)}'))
                
    if use_wandb == True:
        wandb.log({'gen_loss_val': mean_loss_gen/len(data_loader),
                   'disc_loss_val': mean_loss_disc/len(data_loader)})

    return (mean_loss_gen/len(data_loader)), (mean_loss_disc / len(data_loader))


def valid_on_the_fly(gen, disc, data_loader,epoch,save_dir,device):

    gen.eval()
    disc.eval()

    with torch.no_grad():
        for batch in data_loader:
            input_img_batch = batch[0]
            input_mask_batch = batch[1]
            
            input_img = input_img_batch[:1,:,:,:].to(device)
            input_mask = input_mask_batch[:1,:,:,:].to(device)
           

            gen_img = gen(input_mask)
            ans_gen = disc(input_mask,gen_img)
            break

        plt_save_example_synth_img(input_img_ref=input_img[0,0,:,:].detach().cpu().numpy(), 
                                    input_mask_ref=input_mask[0,0,:,:].detach().cpu().numpy(), 
                                    gen_img_ref=gen_img[0,0,:,:].detach().cpu().numpy(), 
                                    disc_ans=ans_gen[0,0,:,:].detach().cpu().numpy(), 
                                    epoch=epoch+1, 
                                    save_dir=save_dir)
            




