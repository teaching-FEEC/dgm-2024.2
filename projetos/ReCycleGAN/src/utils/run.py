"""Functions to control training and testing CycleGAN models."""
import time
import gc
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import get_gpu_memory_usage


def save_losses(loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id, loss_G_plp, filename='losses.txt'):
    """
    Saves the generator and discriminator losses to a text file.

    Saves a text file containing the losses for the generator and discriminators
    (A and B) over the training epochs.

    Args:
    - loss_G (list): List of generator losses over the training epochs.
    - loss_D_A (list): List of discriminator A losses over the training epochs.
    - loss_D_B (list): List of discriminator B losses over the training epochs.
    - filename (str): The file path where the losses will be saved. Defaults to 'losses.txt'.
    """
    np.savetxt(
        filename,
        np.column_stack((loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id, loss_G_plp)),
        header='Generator total loss, Discriminator A loss, Discriminator B loss')

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

    loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id, loss_G_plp = 0, 0, 0, 0, 0, 0, 0
    for batch_A, batch_B in progress_bar:
        progress_bar.set_description(f'Epoch {epoch:03d}')

        if n_samples is not None:
            batch_A = batch_A[:n_samples]
            batch_B = batch_B[:n_samples]

        real_A = batch_A.to(device)
        real_B = batch_B.to(device)

        # Perform one optimization step
        loss = model.optimize(real_A, real_B)
        loss_G += loss.loss_G.item()
        loss_D_A += loss.loss_D_A.item()
        loss_D_B += loss.loss_D_B.item()
        loss_G_ad += loss.loss_G_ad.item()
        loss_G_cycle += loss.loss_G_cycle.item()
        loss_G_id += loss.loss_G_id.item()
        loss_G_plp += loss.loss_G_plp.item()

        progress_bar.set_postfix({
            'G_loss': f'{loss.loss_G.item():.4f}',
            'D_A_loss': f'{loss.loss_D_A.item():.4f}',
            'D_B_loss': f'{loss.loss_D_B.item():.4f}',
            'GPU': f'{get_gpu_memory_usage("",True)}',
        })

        torch.cuda.empty_cache()
        gc.collect()

    progress_bar.close()

    loss_G /= (len(train_A) + len(train_B)) / 2
    loss_D_A /= len(train_A)
    loss_D_B /= len(train_B)
    loss_G_ad /= (len(train_A) + len(train_B)) / 2
    loss_G_cycle /= (len(train_A) + len(train_B)) / 2
    loss_G_id /= (len(train_A) + len(train_B)) / 2
    loss_G_plp /= (len(train_A) + len(train_B)) / 2 * plp_step

    msg = f'Epoch {epoch:03d}: G_loss={loss_G:.4g}, '
    msg += f'D_A_loss={loss_D_A:.4g}, D_B_loss={loss_D_B:.4g}, '
    msg += f'G_ad={loss_G_ad:.4g}, G_cycle={loss_G_cycle:.4g}, '
    msg += f'G_id={loss_G_id:.4g}, G_plp={loss_G_plp:.4g}, '
    msg += f'Time={time.time() - time_start:.2f} s'
    print(msg)
    return loss_G, loss_D_A, loss_D_B, loss_G_ad, loss_G_cycle, loss_G_id, loss_G_plp

def plot_losses(train_losses, val_losses):
    """
    Plots the training and validation losses over the epochs.

    Args:
    - train_losses (list): List of training losses (e.g., generator losses) over the epochs.
    - val_losses (list): List of validation losses over the epochs.

    Displays:
    - A line plot showing the progression of training and validation losses.
    - Training and validation losses are plotted against the number of epochs.
    """
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label='Training Loss',
        linewidth=2, alpha=0.7)
    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label='Validation Loss',
        linewidth=2, alpha=0.7)
    plt.title('CycleGAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()