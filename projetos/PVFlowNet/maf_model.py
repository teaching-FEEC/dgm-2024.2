import torch
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform
from nflows.flows import Flow

class MAFModel(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_layers, condition_size):
        super(MAFModel, self).__init__()

        self.base_distribution = StandardNormal([num_inputs])

        transforms = []
        for _ in range(num_layers):
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_inputs, hidden_features=num_hidden, context_features=condition_size))

        transform = CompositeTransform(transforms)

        self.flow = Flow(transform=transform, distribution=self.base_distribution)

    def forward(self, x, condition):
        return self.flow.log_prob(x, context=condition)

    def sample(self, num_samples, condition):
        return self.flow.sample(num_samples, context=condition)
