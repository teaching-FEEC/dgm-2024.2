import torch
import torch.nn as nn

class GeneratorPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GeneratorPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.model(state)
