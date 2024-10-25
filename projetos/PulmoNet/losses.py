import torch
import torch.nn as nn
from utils import check_for_zero_loss


def get_regularization(regularizer_type, input_mask, input_img, gen_img):
    if regularizer_type == 'MAE':
        return check_for_zero_loss(torch.sum(torch.abs((input_img-gen_img)), dim=(1, 2)).mean())
    elif regularizer_type == 'MSE':
        return check_for_zero_loss(torch.sum(((input_img-gen_img)**2), dim=(1, 2)).mean())
    elif regularizer_type == 'RMSE':
        return check_for_zero_loss(torch.sqrt(torch.sum(((input_img-gen_img)**2), dim=(1, 2)).mean()))
    elif regularizer_type == 'MAE_mask':
        return check_for_zero_loss(torch.sum(torch.abs(input_mask*(input_img-gen_img)), dim=(1, 2)).mean())
    elif regularizer_type == 'MSE_mask':
        return check_for_zero_loss(torch.sum(((input_mask*(input_img-gen_img))**2), dim=(1, 2)).mean())
    elif regularizer_type == 'RMSE_mask':
        return check_for_zero_loss(torch.sqrt(torch.sum(((input_mask*(input_img-gen_img))**2), dim=(1, 2)).mean()))
    elif regularizer_type == 'MAE_outside_mask':
        return check_for_zero_loss(torch.sum(torch.abs((1-input_mask)*(input_img-gen_img)), dim=(1, 2)).mean())
    elif regularizer_type == 'MSE_outside_mask':
        return check_for_zero_loss(torch.sum((((1-input_mask)*(input_img-gen_img))**2), dim=(1, 2)).mean())
    elif regularizer_type == 'RMSE_outside_mask':
        return check_for_zero_loss(torch.sqrt(torch.sum((((1-input_mask)*(input_img-gen_img))**2), dim=(1, 2)).mean()))
    else:
        raise ValueError(f"Invalid regularizer_type: {regularizer_type}, check losses.py and add the desired regularization.")


def get_disc_loss(gen, disc, criterion, input_mask, input_img, device):
    gen_img = gen(input_mask).detach()
    ans_gen = disc(gen_img)
    gt_gen = torch.zeros_like(ans_gen)
    ans_real = disc(input_img)
    gt_real = torch.ones_like(ans_real)
    # Concatenando os vetores do output do discriminador das reais com as geradas
    x = torch.cat((ans_real.reshape(-1), ans_gen.reshape(-1)))
    # Concatenando os vetores dos labels reais das images reais com as geradas
    y = torch.cat((gt_real.reshape(-1), gt_gen.reshape(-1)))
    loss = criterion(x, y)
    # The regularization (l1 norm) is not important here: is independent of D
    return loss.mean()


def get_gen_loss(gen, disc, criterion, input_mask, input_img, device, regularization_type=None, regularization_level=None):
    gen_img = gen(input_mask)
    ans_gen = disc(gen_img)
    # we want ans_gen close to 1: to trick the disc
    gt_gen = torch.ones_like(ans_gen)
    loss = criterion(ans_gen, gt_gen).mean()
    if regularization_type is not None:
        if isinstance(regularization_type, list):
            for idx, regularization in enumerate(regularization_type):
                loss = loss + regularization_level[idx]*get_regularization(regularizer_type=regularization_type[idx], input_mask=input_mask, input_img=input_img, gen_img=gen_img)
        else:
            loss = loss + regularization_level*get_regularization(regularizer_type=regularization_type, input_mask=input_mask, input_img=input_img, gen_img=gen_img)
    return loss
