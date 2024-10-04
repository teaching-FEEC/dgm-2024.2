import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import os

class Encoder(nn.Module):
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block='LSTM'):
        super(Encoder, self).__init__()
        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout=dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout=dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        _, (h_end, _) = self.model(x)
        h_end = h_end[-1, :, :]
        return h_end

class Lambda(nn.Module):
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()
        self.hidden_to_mean = nn.Linear(hidden_size, latent_length)
        self.hidden_to_logvar = nn.Linear(hidden_size, latent_length)
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        latent_mean = self.hidden_to_mean(cell_output)
        latent_logvar = self.hidden_to_logvar(cell_output)
        if self.training:
            std = torch.exp(0.5 * latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(latent_mean)
        else:
            return latent_mean

class Decoder(nn.Module):
    def __init__(self, sequence_length, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(latent_length, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.decoder_inputs = torch.zeros(sequence_length, 1, requires_grad=True).type(dtype)

    def forward(self, latent):
        h_state = self.latent_to_hidden(latent).unsqueeze(0)  # Add a sequence dimension
        decoder_output, _ = self.model(self.decoder_inputs.unsqueeze(1), h_state)
        out = self.hidden_to_output(decoder_output)
        return out

class VRAE(nn.Module):
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM', n_epochs=5, dropout_rate=0.,
                 optimizer='Adam', loss='MSELoss', print_every=100, clip=True, max_grad_norm=5, dload='.'):
        super(VRAE, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.encoder = Encoder(number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout_rate, block).to(self.device)
        self.lmbd = Lambda(hidden_size, latent_length).to(self.device)
        self.decoder = Decoder(sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, number_of_features, self.dtype, block).to(self.device)

        # Inicializando otimizador e função de perda
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

        if loss == 'MSELoss':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Unsupported loss function")

        self.n_epochs = n_epochs

    def _create_dataset(self, x, y):
        return [(torch.tensor(x[i], dtype=self.dtype).to(self.device), torch.tensor(y[i], dtype=torch.float32).to(self.device)) for i in range(len(y))]

    def fit(self, x_train, y_train, save=False):
        dataset = self._create_dataset(x_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(self.n_epochs):
            self.train()
            total_loss = 0

            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                encoded = self.encoder(x_batch)
                latent = self.lmbd(encoded)
                decoded = self.decoder(latent)

                # Calculando a perda
                loss = self.loss_fn(decoded, x_batch)  # Aqui você pode ajustar a perda conforme necessário
                loss.backward()
                
                if clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch + 1}/{self.n_epochs}], Loss: {avg_loss:.4f}')

            if save:
                # Lógica para salvar o modelo
                pass
