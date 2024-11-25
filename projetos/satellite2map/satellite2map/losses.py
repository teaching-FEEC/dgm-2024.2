import torch
import torch.nn as nn

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha=alpha
        self.bce=nn.MSELoss()
        self.l1=nn.L1Loss()
        
    def forward(self, fake, real, fake_pred):
        fake_target = torch.ones_like(fake_pred)
        loss = self.bce(fake_pred, fake_target) + self.alpha * self.l1(fake, real)
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
    
class GenCGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, fake_pred):
        fake_target = torch.ones_like(fake_pred)
        loss = self.mse(fake_pred, fake_target)
        return loss
    
class DiscCGANLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.mse(fake_pred, fake_target)
        real_loss = self.mse(real_pred, real_target)
        loss = (fake_loss + real_loss)/2
        return loss
    
class CycleConsistencyLoss(nn.Module):
    def __init__(self, alpha=10.0):
        super().__init__()
        self.cc_loss = nn.L1Loss()
        self.alpha = alpha
        
    def forward(self, x, x_pred):
        return self.alpha * self.cc_loss(x, x_pred)
    
class IdentityLoss(nn.Module):
    def __init__(self, alpha=5.0):
        super().__init__()
        self.id_loss = nn.L1Loss()
        self.alpha = alpha
        
    def forward(self, same_img, real_img):
        return self.alpha * self.id_loss(same_img, real_img)