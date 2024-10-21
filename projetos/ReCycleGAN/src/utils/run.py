# pylint: disable=C0103
"""Functions to control training and testing CycleGAN models."""
import time
import gc
from dataclasses import dataclass
import pandas as pd
import torch
from tqdm import tqdm
from .utils import get_gpu_memory_usage

@dataclass
class Loss:
    """Dataclass for CycleGAN losses as torch.Tensor."""
    loss_G: torch.Tensor
    loss_D_A: torch.Tensor
    loss_D_B: torch.Tensor
    loss_G_ad: torch.Tensor
    loss_G_cycle: torch.Tensor
    loss_G_id: torch.Tensor
    loss_G_plp: torch.Tensor

class LossValues:
    """Class for CycleGAN losses as float.

    Args:
    - n_A (int): Number of samples in domain A.
    - n_B (int): Number of samples in domain B.
    - plp_step (int): Steps between Path Length Penalty calculations.
    """
    def __init__(self, n_A, n_B, plp_step):
        self.loss_G = 0
        self.loss_D_A = 0
        self.loss_D_B = 0
        self.loss_G_ad = 0
        self.loss_G_cycle = 0
        self.loss_G_id = 0
        self.loss_G_plp = 0

        self.n_a = n_A
        self.n_b = n_B
        self.plp_step = plp_step

    def add(self, loss: Loss):
        """Adds the losses to the respective values."""
        self.loss_G += loss.loss_G.item()
        self.loss_D_A += loss.loss_D_A.item()
        self.loss_D_B += loss.loss_D_B.item()
        self.loss_G_ad += loss.loss_G_ad.item()
        self.loss_G_cycle += loss.loss_G_cycle.item()
        self.loss_G_id += loss.loss_G_id.item()
        self.loss_G_plp += loss.loss_G_plp.item()

    def normalize(self):
        """Normalizes the losses by the number of samples."""
        self.loss_G /= (self.n_a + self.n_b)/2
        self.loss_D_A /= self.n_a
        self.loss_D_B /= self.n_b
        self.loss_G_ad /= (self.n_a + self.n_b)/2
        self.loss_G_cycle /= (self.n_a + self.n_b)/2
        self.loss_G_id /= (self.n_a + self.n_b)/2
        self.loss_G_plp /= (self.n_a + self.n_b)/2 * self.plp_step

    def __str__(self):
        out = []
        out.append(f'G_loss={self.loss_G:.4g}')
        out.append(f'D_A_loss={self.loss_D_A:.4g}')
        out.append(f'D_B_loss={self.loss_D_B:.4g}')
        out.append(f'G_ad={self.loss_G_ad:.4g}')
        out.append(f'G_cycle={self.loss_G_cycle:.4g}')
        out.append(f'G_id={self.loss_G_id:.4g}')
        out.append(f'G_plp={self.loss_G_plp:.4g}')
        return ', '.join(out)

class LossLists:
    """Class for CycleGAN losses lists."""
    def __init__(self):
        self.loss_G = []
        self.loss_D_A = []
        self.loss_D_B = []
        self.loss_G_ad = []
        self.loss_G_cycle = []
        self.loss_G_id = []
        self.loss_G_plp = []

    def append(self, loss: Loss):
        """Appends the losses to the respective lists."""
        self.loss_G.append(loss.loss_G)
        self.loss_D_A.append(loss.loss_D_A)
        self.loss_D_B.append(loss.loss_D_B)
        self.loss_G_ad.append(loss.loss_G_ad)
        self.loss_G_cycle.append(loss.loss_G_cycle)
        self.loss_G_id.append(loss.loss_G_id)
        self.loss_G_plp.append(loss.loss_G_plp)

    def names(self):
        """Returns the names of the losses."""
        return ['G_loss', 'D_A_loss', 'D_B_loss', 'G_ad', 'G_cycle', 'G_id', 'G_plp']

    def to_dict(self):
        """Returns the losses as a dictionary."""
        return {
            'G_loss': self.loss_G,
            'D_A_loss': self.loss_D_A,
            'D_B_loss': self.loss_D_B,
            'G_ad': self.loss_G_ad,
            'G_cycle': self.loss_G_cycle,
            'G_id': self.loss_G_id,
            'G_plp': self.loss_G_plp
        }

    def to_dataframe(self):
        """Returns the losses as a pandas DataFrame."""
        return pd.DataFrame(self.to_dict())


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

    losses_ = LossValues(len(train_A.dataset), len(train_B.dataset), plp_step)
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
