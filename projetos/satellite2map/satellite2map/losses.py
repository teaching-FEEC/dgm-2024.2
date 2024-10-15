import torch
import torch.nn as nn

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha=alpha
        self.bce=nn.BCEWithLogitsLoss()
        self.l1=nn.L1Loss()
        
    def forward(self, fake, real, fake_pred, use_l1=True):
        use_l1 = 1 if use_l1 else 0
        fake_target = torch.ones_like(fake_pred)
        loss = self.bce(fake_pred, fake_target) + self.alpha * self.l1(fake, real) * use_l1
        return loss
    
class DiscriminatorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.loss_fn(fake_pred, fake_target)
        real_loss = self.loss_fn(real_pred, real_target)
        loss = (fake_loss + real_loss)/2
        return loss
    
class CycleConsistencyLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.cc_loss = nn.L1Loss()
        
    def forward(self, x, x_pred, y, y_pred):
        return self.cc_loss(x, x_pred) + self.cc_loss(y, y_pred)