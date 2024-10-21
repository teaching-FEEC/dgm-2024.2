# pylint: disable=C0103
"""Module with CycleGAN losses."""
from dataclasses import dataclass
import pandas as pd
import torch

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
    - n_A (int): Number of batches in domain A.
    - n_B (int): Number of batches in domain B.
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
