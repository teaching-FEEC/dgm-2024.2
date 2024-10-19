# modified from Github repo: https://github.com/MICLab-Unicamp/Spectro-ViT mainteined by Gabriel Dias (g172441@dac.unicamp.br), Mateus Oliveira (m203656@dac.unicamp.br)

import torch
import torch.optim.lr_scheduler as lr_scheduler

class LRScheduler:
    def __init__(self,
                 optimizer,
                 scheduler_type,
                 **kwargs):

        self.scheduler_type = scheduler_type
        self.scheduler = self._create_scheduler(optimizer, **kwargs)

    def _create_scheduler(self, optimizer, **kwargs):
        if self.scheduler_type == 'LinearLR':
            return lr_scheduler.LinearLR(optimizer, **kwargs)
        
        elif self.scheduler_type == 'StepLR':
            return lr_scheduler.StepLR(optimizer, **kwargs)

        elif self.scheduler_type == 'MultiStepLR':
            return lr_scheduler.MultiStepLR(optimizer, **kwargs)

        elif self.scheduler_type == 'ExponentialLR':
            return lr_scheduler.ExponentialLR(optimizer, **kwargs)

        elif self.scheduler_type == 'CosineAnnealingLR':
            return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

        else:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}, check lr_scheduler_and_optim.py and add the desired scheduler type.")

    def step(self, *args, **kwargs):
        self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


def get_optimizer(model, optimizer_type, learning_rate, **kwargs):
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(),lr=learning_rate, **kwargs)
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(model.parameters(),lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Invalid optimizer_type: {optimizer_type}, check lr_scheduler_and_optim.py and add the desired criterion.")
