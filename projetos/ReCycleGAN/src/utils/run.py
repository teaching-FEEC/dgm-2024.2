# pylint: disable=C0103,E0401,C0411
"""Functions to control training and testing CycleGAN models."""
import time
import gc

import torch
from torchvision import transforms
from tqdm import tqdm
import wandb

from src.utils import get_gpu_memory_usage, get_current_commit
from src.utils.data_loader import get_img_dataloader
from src.models.cyclegan import CycleGAN
from src.models.losses import LossValues, LossLists


def save_losses(loss: LossLists, filename='losses.txt'):
    """
    Saves the generator and discriminator losses to a text file.

    Saves a text file containing the losses for the generator and discriminators
    (A and B) over the training epochs.

    Args:
    - loss (LossLists): An instance of LossLists containing lists of losses.
    - filename (str): The file path where the losses will be saved. Defaults to 'losses.txt'.
    """
    df = loss.to_dataframe()
    df = df.rename_axis('Epoch')
    df.to_csv(filename, index=True)


def train_one_epoch(epoch, model, train_A, train_B, device, n_samples=None, plp_step=0):
    """
    Trains the CycleGAN model for a single epoch and returns the generator and discriminator losses.

    Args:
    - epoch (int): The current epoch number.
    - model (CycleGAN): The CycleGAN model instance.
    - train_A (DataLoader): DataLoader for domain A training images.
    - train_B (DataLoader): DataLoader for domain B training images.
    - device (torch.device): The device on which the model and data are
    loaded (e.g., 'cuda' or 'cpu').
    - n_samples (int): Number of samples to train on per batch.
    If None, train on all samples. Default is None.
    - plp_step: Steps between Path Length Penalty calculations. Used to adjust
    PLP loss value. Default is 0.

    Returns:
    - loss_G (float): The total loss of the generator for this epoch.
    - loss_D_A (float): The total loss of discriminator A for this epoch.
    - loss_D_B (float): The total loss of discriminator B for this epoch.

    During training:
    - It iterates through the batches of both domains (A and B) and performs
    optimization on the generators and discriminators.
    - Progress is tracked with a `tqdm` progress bar that shows current generator
    and discriminator losses.
    """
    time_start = time.time()
    progress_bar = tqdm(zip(train_A, train_B), desc=f'Epoch {epoch:03d}',
                        leave=False, disable=False)

    losses_ = LossValues(len(train_A), len(train_B), plp_step)
    for batch_A, batch_B in progress_bar:
        progress_bar.set_description(f'Epoch {epoch:03d}')

        if n_samples is not None:
            batch_A = batch_A[:n_samples]
            batch_B = batch_B[:n_samples]

        real_A = batch_A.to(device)
        real_B = batch_B.to(device)

        loss = model.optimize(real_A, real_B)

        losses_.add(loss)

        progress_bar.set_postfix({
            'G_loss': f'{loss.loss_G.item():.4f}',
            'D_A_loss': f'{loss.loss_D_A.item():.4f}',
            'D_B_loss': f'{loss.loss_D_B.item():.4f}',
            'GPU': f'{get_gpu_memory_usage("",True)}',
        })

        torch.cuda.empty_cache()
        gc.collect()

    progress_bar.close()
    losses_.normalize()
    print(f'Epoch {epoch:03d}: {str(losses_)}, Time={time.time() - time_start:.2f} s')
    return losses_
