    
from tqdm import trange
import torch
import gc
from losses import *
from utils import plt_save_example_synth_img, plt_save_example_synth_img_with_airway, set_requires_grad, plt_save_example_airways_img
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
                    regularization_level=None,
                    generate_airway_segmentation=False):

    mean_loss_gen = 0
    mean_loss_disc = 0

    counter_batches_used_to_upd_disc = 0
    counter_batches_used_to_upd_gen = 0

    counter_steps_before_upd_disc = 0
    counter_steps_before_upd_gen = 0
    

    with trange(len(data_loader), desc='Train Loop') as progress_bar:
        for batch_idx, batch in zip(progress_bar, data_loader):

            input_img_batch = batch[0]
            input_img = input_img_batch.to(device)

            if generate_airway_segmentation is False:
                input_mask_batch = batch[1]
                input_mask = input_mask_batch.to(device)
            else:
                input_airway_batch = batch[1]
                input_mask_batch = batch[2]
                input_airway = input_airway_batch.to(device)
                input_mask = input_mask_batch.to(device)
            
            if counter_steps_before_upd_disc == 0:
                set_requires_grad(model=disc,set_require_grad=True)
                disc_opt.zero_grad()
                if generate_airway_segmentation is False:
                    disc_loss = get_disc_loss(gen=gen,
                                              disc=disc,
                                              criterion=criterion,
                                              input_mask=input_mask,
                                              input_img=input_img)
                else:
                    disc_loss = get_disc_loss_airwaygen(gen=gen, 
                                                        disc=disc,
                                                        criterion=criterion,
                                                        input_mask=input_mask,
                                                        input_img=input_img,
                                                        input_airway=input_airway)

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
                if generate_airway_segmentation is False:
                    gen_loss = get_gen_loss(gen=gen,
                                            disc=disc,
                                            criterion=criterion,
                                            input_mask=input_mask,
                                            input_img=input_img,
                                            regularizer=regularizer,
                                            regularization_level=regularization_level,
                                            device=device)
                else:
                    gen_loss = get_gen_loss_airwaygen(gen=gen,
                                                    disc=disc, 
                                                    criterion=criterion, 
                                                    input_mask=input_mask, 
                                                    input_img=input_img, 
                                                    input_airway=input_airway, 
                                                    device=device, 
                                                    regularizer=regularizer, 
                                                    regularization_level=regularization_level)
                gen_loss.backward(retain_graph=True)
                gen_opt.step()
                mean_loss_gen = mean_loss_gen + gen_loss.item() 
                counter_batches_used_to_upd_gen = counter_batches_used_to_upd_gen + 1
                counter_steps_before_upd_disc = counter_steps_before_upd_disc + 1
                if counter_steps_before_upd_disc == steps_to_complete_bfr_upd_disc:
                    counter_steps_before_upd_disc = 0

            if counter_batches_used_to_upd_disc > 0 and counter_batches_used_to_upd_gen > 0:
                progress_bar.set_postfix(
                    desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(data_loader):d},'
                        f'gen loss: {mean_loss_gen / (counter_batches_used_to_upd_gen)},'
                        f'disc loss: {mean_loss_disc / (counter_batches_used_to_upd_disc)}'))
            elif counter_batches_used_to_upd_disc > 0 and counter_batches_used_to_upd_gen == 0:
                progress_bar.set_postfix(
                    desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(data_loader):d},'
                        f'disc loss: {mean_loss_disc / (counter_batches_used_to_upd_disc)}'))
            elif counter_batches_used_to_upd_disc == 0 and counter_batches_used_to_upd_gen > 0:
                progress_bar.set_postfix(
                    desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(data_loader):d},'
                        f'gen loss: {mean_loss_gen / (counter_batches_used_to_upd_gen)}'))


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
                         regularization_level=None,
                         generate_airway_segmentation=False):

    mean_loss_gen = 0
    mean_loss_disc = 0

    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
        with trange(len(data_loader), desc='Validation Loop') as progress_bar:
            for batch_idx, batch in zip(progress_bar, data_loader):

                input_img_batch = batch[0]
                input_img = input_img_batch.to(device)

                if generate_airway_segmentation is False:
                    input_mask_batch = batch[1]
                    input_mask = input_mask_batch.to(device)
                else:
                    input_airway_batch = batch[1]
                    input_mask_batch = batch[2]
                    input_airway = input_airway_batch.to(device)
                    input_mask = input_mask_batch.to(device)

                if generate_airway_segmentation is False:   
                    disc_loss = get_disc_loss(gen=gen,
                                              disc=disc,
                                              criterion=criterion,
                                              input_mask=input_mask,
                                              input_img=input_img)
                else:
                    disc_loss = get_disc_loss_airwaygen(gen=gen, 
                                                        disc=disc,
                                                        criterion=criterion,
                                                        input_mask=input_mask,
                                                        input_img=input_img,
                                                        input_airway=input_airway)
                mean_loss_disc = mean_loss_disc + disc_loss.item() 
                if generate_airway_segmentation is False:  
                    gen_loss = get_gen_loss(gen=gen,
                                            disc=disc,
                                            criterion=criterion,
                                            input_mask=input_mask,
                                            input_img=input_img,
                                            regularizer=regularizer,
                                            regularization_level=regularization_level,
                                            device=device)
                else:
                    gen_loss = get_gen_loss_airwaygen(gen=gen,
                                                  disc=disc, 
                                                  criterion=criterion, 
                                                  input_mask=input_mask, 
                                                  input_img=input_img, 
                                                  input_airway=input_airway, 
                                                  device=device, 
                                                  regularizer=regularizer, 
                                                  regularization_level=regularization_level)
                mean_loss_gen = mean_loss_gen + gen_loss.item() 

                progress_bar.set_postfix(
                desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(data_loader):d},'
                      f'gen loss: {mean_loss_gen / (batch_idx + 1)},'
                      f'disc loss: {mean_loss_disc / (batch_idx + 1)}'))
                
    if use_wandb == True:
        wandb.log({'gen_loss_val': mean_loss_gen/len(data_loader),
                   'disc_loss_val': mean_loss_disc/len(data_loader)})

    return (mean_loss_gen/len(data_loader)), (mean_loss_disc / len(data_loader))


def valid_on_the_fly(gen, disc, data_loader,epoch,save_dir,device, generate_airway_segmentation=False):

    gen.eval()
    disc.eval()

    with torch.no_grad():
        for batch in data_loader:
            input_img_batch = batch[0]
            input_img = input_img_batch[:1,:,:,:].to(device)
            if generate_airway_segmentation is False:
                input_mask_batch = batch[1]
                input_mask = input_mask_batch[:1,:,:,:].to(device)
            else:
                input_airway_batch = batch[1]
                input_mask_batch = batch[2]
                input_airway = input_airway_batch[:1,:,:,:].to(device)
                input_mask = input_mask_batch[:1,:,:,:].to(device)
            
            gen_img = gen(input_mask)
            ans_gen = disc(input_mask,gen_img)
            break


        if generate_airway_segmentation is False:
            plt_save_example_synth_img(input_img_ref=input_img[0,0,:,:].detach().cpu().numpy(), 
                                        input_mask_ref=input_mask[0,0,:,:].detach().cpu().numpy(), 
                                        gen_img_ref=gen_img[0,0,:,:].detach().cpu().numpy(), 
                                        disc_ans=ans_gen[0,0,:,:].detach().cpu().numpy(), 
                                        epoch=epoch+1, 
                                        save_dir=save_dir)
        
        else:
            plt_save_example_synth_img_with_airway(input_img_ref=input_img[0,0,:,:].detach().cpu().numpy(),
                                                    input_mask_ref=input_mask[0,0,:,:].detach().cpu().numpy(), 
                                                    input_airway_ref=input_airway[0,0,:,:].detach().cpu().numpy(), 
                                                    gen_img_ref=gen_img[0,0,:,:].detach().cpu().numpy(), 
                                                    gen_airway_ref=gen_img[0,1,:,:].detach().cpu().numpy(), 
                                                    disc_ans=ans_gen[0,0,:,:].detach().cpu().numpy(), 
                                                    epoch=epoch+1, 
                                                    save_dir=save_dir)
            
### U-Net ---------------------------------------------------------
def run_train_epoch_unet(unet, criterion, data_loader, unet_opt, epoch, device):

    mean_loss = 0 
    counter_batches_used_to_upd = 0   

    with trange(len(data_loader), desc='Train Loop') as progress_bar:
        for batch_idx, batch in zip(progress_bar, data_loader):

            input_img_batch = batch[0]
            input_airway_batch = batch[1]

            input_img = input_img_batch.to(device)
            input_airway = input_airway_batch.to(device)
            
            unet_opt.zero_grad()
            loss = get_unet_loss(unet,criterion,input_airway,input_img,device)
            if loss.requires_grad:
                loss.backward()  # Compute gradients
                unet_opt.step()  # Update model parameters
            else:
                print("Loss does not require gradients. Check your input tensors.")
            mean_loss = mean_loss + loss.item() 
            counter_batches_used_to_upd += 1

            progress_bar.set_postfix(
                desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(data_loader):d},'
                      f'loss: {(mean_loss / counter_batches_used_to_upd)}'))
    
    return (mean_loss / counter_batches_used_to_upd)


def run_validation_epoch_unet(unet, criterion, data_loader, epoch, device):

    mean_loss = 0

    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
        with trange(len(data_loader), desc='Validation Loop') as progress_bar:
            for batch_idx, batch in zip(progress_bar, data_loader):

                input_img_batch = batch[0]
                input_airway_batch = batch[1]

                input_img = input_img_batch.to(device)
                input_airway = input_airway_batch.to(device)
                loss = get_unet_loss(unet,criterion,input_airway,input_img,device)
                mean_loss = mean_loss + loss.item() 

                progress_bar.set_postfix(
                desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(data_loader):d},'
                      f'loss: {mean_loss / (batch_idx + 1)}'))

    return (mean_loss/len(data_loader))


def valid_on_the_fly_unet(unet, data_loader,epoch,save_dir,device):

    unet.eval()

    with torch.no_grad():
        for batch in data_loader:
            input_img_batch = batch[0]
            input_airway_batch = batch[1]
            
            input_img = input_img_batch[:1,:,:,:].to(device)
            input_airway = input_airway_batch[:1,:,:,:].to(device)
           
            gen_seg = unet(input_img)
            break

        plt_save_example_airways_img(input_img_ref=input_img[0,0,:,:].detach().cpu().numpy(), 
                                    input_airway_ref=input_airway[0,0,:,:].detach().cpu().numpy(), 
                                    gen_seg_ref=gen_seg[0,0,:,:].detach().cpu().numpy(), 
                                    epoch=epoch+1, 
                                    save_dir=save_dir)