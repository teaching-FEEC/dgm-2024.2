import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Discriminator, self).__init__()
        input_dim = state_dim + action_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.model(x)
