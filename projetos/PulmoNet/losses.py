#Implementing losses functions for PulmoNet generator and discriminator
#Accounts for a regularization for the generator loss

from typing import Any
import torch
import torch.nn as nn
import numpy as np

#Notation: input_mask: mask  // input_img: original image // gen_img: synthetic image
#           input_airway: airway annotation

#Accepts different types of regularization that can be combined if regularization_type is a list
#Regularization level can be change at each call
class Regularizer:
    def __init__(self, regularization_type):
        self.L1norm = None
        self.L2norm = None
        if regularization_type is not None:
            if not isinstance(regularization_type, list):
                self.regularization_type = [regularization_type]
            else:
                self.regularization_type = regularization_type
                
            for reg in self.regularization_type:
                if 'MAE' in reg:
                    self.L1norm = nn.L1Loss()
                elif 'MSE' in reg:
                    self.L2norm = nn.MSELoss()
        else:
            self.regularization_type = None
    
    #if 'mask' in regularization_type: regularizes only the masked region
    #if 'outside_mask' in regularization_type: regularizes only the region outside mask
    #if none of the above: applied to the whole image
    #accept both L1 and L2 regularization
    #accepts list so multiple regularizations can be applied at different levels
    def __call__(self, regularization_level, input_mask, input_img, gen_img, device):
        if self.regularization_type is not None:
            if not isinstance(regularization_level, list):
                regularization_level = [regularization_level]

            regularization = torch.tensor(0,dtype=torch.float32).to(device)
            for idx,reg in enumerate(self.regularization_type):
                if reg == 'MAE':
                    regularization += regularization_level[idx]*self.L1norm(gen_img,input_img)
                elif reg == 'MSE':
                    regularization += regularization_level[idx]*self.L2norm(gen_img,input_img)
                elif reg == 'MAE_mask':
                    regularization += regularization_level[idx]*self.L1norm(input_mask*gen_img,
                                                                            input_mask*input_img)
                elif reg == 'MSE_mask':
                    regularization += regularization_level[idx]*self.L2norm(input_mask*gen_img,
                                                                            input_mask*input_img)
                elif reg == 'MAE_outside_mask':
                    regularization += regularization_level[idx]*self.L1norm((1-input_mask)*gen_img,
                                                                            (1-input_mask)*input_img)
                elif reg == 'MSE_outside_mask':
                    regularization += regularization_level[idx]*self.L2norm((1-input_mask)*gen_img,
                                                                            (1-input_mask)*input_img)
                else:
                    raise ValueError(f"Invalid regularizer_type: {reg}, check losses.py and add the desired regularization.")
            return regularization
        else:
            return torch.tensor(0)


#-----------------------------------------Loss for CT image synthesis-----------------------------

#For Discriminator's loss: average between discriminator answer for real and synthetic images
#real: 1
#fake: 0
def get_disc_loss(gen, disc, criterion, input_mask, input_img):
    gen_img = gen(input_mask).detach()
    ans_gen = disc(input_mask,gen_img)
    gt_gen = torch.zeros_like(ans_gen)
    ans_real = disc(input_mask,input_img)
    gt_real = torch.ones_like(ans_real)

    loss_fake = criterion(ans_gen,gt_gen)
    loss_real = criterion(ans_real,gt_real)
    loss = (loss_fake+loss_real)*0.5
    return loss

#For Generator's loss: accepts regularization using an instance of Regularizer class (regularizer)
#regularization level can be a scalar or a list
#we don't tell the criterion that the images are fake --> we want disc to output 1, because the  generator tricked it
def get_gen_loss(gen, disc, criterion, input_mask, input_img, device, 
                 regularizer=None, regularization_level=None):
    gen_img = gen(input_mask)
    ans_gen = disc(input_mask,gen_img)
    # we want ans_gen close to 1: to trick the disc
    gt_gen = torch.ones_like(ans_gen)
    loss = criterion(ans_gen, gt_gen).mean()
    if regularizer is not None:
        regularization = regularizer(regularization_level=regularization_level, 
                                     input_mask=input_mask, 
                                     input_img=input_img, gen_img=gen_img, 
                                     device=device)
        loss += regularization
    return loss

#------------------------------Loss for CT image and airway segmentation synthesis-----------------------------

def get_disc_loss_airwaygen(gen, disc, criterion, input_mask, input_img, input_airway):
    gen_img = gen(input_mask).detach()
    ans_gen = disc(input_mask,gen_img)
    gt_gen = torch.zeros_like(ans_gen)
    #if we are generating the airway: disc expects 3-channel input: 
    ans_real = disc(input_mask,torch.cat([input_img, input_airway], dim=1))
    gt_real = torch.ones_like(ans_real)
    loss_fake = criterion(ans_gen,gt_gen)
    loss_real = criterion(ans_real,gt_real)
    loss = (loss_fake+loss_real)*0.5
    return loss

def get_gen_loss_airwaygen(gen, disc, criterion, input_mask, input_img, 
                           input_airway, device, regularizer=None, 
                           regularization_level=None):
    gen_img = gen(input_mask)
    ans_gen = disc(input_mask,gen_img)
    gt_gen = torch.ones_like(ans_gen)
    loss = criterion(ans_gen, gt_gen).mean()
    if regularizer is not None:
        #both the CT image and airway need regularization
        #if regularization is at masked region, should be on both
        input_to_reg = torch.cat([input_img, input_airway], dim=1)
        input_mask_aux = torch.cat([input_mask, input_mask], dim=1)
        regularization = regularizer(regularization_level=regularization_level, 
                                     input_mask=input_mask_aux, 
                                     input_img=input_to_reg, 
                                     gen_img=gen_img, device=device)
        loss += regularization
    return loss

#------------------------------Loss for U-Net (segmentation network)-----------------------------
def get_unet_loss(unet,criterion,target,input,device):
    out_unet = unet(input) #U-Net output
    loss = criterion(np.squeeze(out_unet),target)
    return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    #implements DICE metric
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get predictions
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice


