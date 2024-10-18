import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_time(data):
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        temp_time = len(data[i])
        time.append(temp_time)
        if temp_time > max_seq_len:
            max_seq_len = temp_time
    return time, max_seq_len

def rnn_cell(module_name):
    if module_name == 'gru':
        rnn = torch.nn.GRU
    elif module_name == 'lstm':
        rnn = torch.nn.LSTM
    else:
        rnn = torch.nn.RNN
    return rnn

def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    Z_mb = []
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp)
    
    Z_mb = torch.tensor(np.array(Z_mb), dtype=torch.float32)
    return Z_mb


# Gerador de lotes (batches) de dados para o treinamento
def batch_generator(ori_data, ori_time, batch_size):
    no = len(ori_data)
    idx = np.random.permutation(no)[:batch_size]
    X_mb = [ori_data[i] for i in idx]
    T_mb = [ori_time[i] for i in idx]
    return X_mb, T_mb


class TimeGAN:
    def __init__(self, ori_data, parameters):
        # Basic Parameters
        self.ori_data = ori_data
        self.no, self.seq_len, self.dim = np.asarray(ori_data).shape
        self.ori_time, self.max_seq_len = extract_time(ori_data)
        self.max_val = np.max(ori_data)
        self.min_val = np.min(ori_data)
        
        # Network Parameters
        self.hidden_dim = parameters['hidden_dim']
        self.num_layers = parameters['num_layer']
        self.iterations = parameters['iterations']
        self.batch_size = parameters['batch_size']
        self.module_name = parameters['module']
        self.z_dim = self.dim
        self.gamma = 1
        self.rnn_cell = rnn_cell(self.module_name)
        
        # Initialize Networks
        self.embedder = self.Embedder(self.dim, self.hidden_dim, self.num_layers, self.rnn_cell).to(device)
        self.recovery = self.Recovery(self.hidden_dim, self.dim, self.num_layers, self.rnn_cell).to(device)
        self.generator = self.Generator(self.z_dim, self.hidden_dim, self.num_layers, self.rnn_cell).to(device)
        self.supervisor = self.Supervisor(self.hidden_dim, self.num_layers, self.rnn_cell).to(device)
        self.discriminator = self.Discriminator(self.hidden_dim, self.num_layers, self.rnn_cell).to(device)

        # Optimizers
        self.e_optimizer = optim.Adam(self.embedder.parameters())
        self.r_optimizer = optim.Adam(self.recovery.parameters())
        self.d_optimizer = optim.Adam(self.discriminator.parameters())
        self.g_optimizer = optim.Adam(self.generator.parameters())
        self.s_optimizer = optim.Adam(self.supervisor.parameters())

    class Embedder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, rnn_cell):
            super(TimeGAN.Embedder, self).__init__()
            self.rnn = rnn_cell(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            h, _ = self.rnn(x)
            h = self.fc(h)
            h = self.sigmoid(h)
            return h

    class Recovery(nn.Module):
        def __init__(self, hidden_dim, output_dim, num_layers, rnn_cell):
            super(TimeGAN.Recovery, self).__init__()
            self.rnn = rnn_cell(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, h):
            x_tilde, _ = self.rnn(h)
            x_tilde = self.fc(x_tilde)
            x_tilde = self.sigmoid(x_tilde)
            return x_tilde

    class Generator(nn.Module):
        def __init__(self, z_dim, hidden_dim, num_layers, rnn_cell):
            super(TimeGAN.Generator, self).__init__()
            self.rnn = rnn_cell(z_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, z):
            e_hat, _ = self.rnn(z)
            e_hat = self.fc(e_hat)
            e_hat = self.sigmoid(e_hat)
            return e_hat

    class Supervisor(nn.Module):
        def __init__(self, hidden_dim, num_layers, rnn_cell):
            super(TimeGAN.Supervisor, self).__init__()
            if num_layers < 1:
                raise ValueError("num_layers must be at least 1")
            self.rnn = rnn_cell(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, h):
            s, _ = self.rnn(h)
            s = self.fc(s)
            s = self.sigmoid(s)
            return s

    class Discriminator(nn.Module):
        def __init__(self, hidden_dim, num_layers, rnn_cell):
            super(TimeGAN.Discriminator, self).__init__()
            self.rnn = rnn_cell(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, h):
            y_hat, _ = self.rnn(h)
            y_hat = self.fc(y_hat)
            return y_hat

    def MinMaxScaler(self, data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = (data - min_val) / (max_val - min_val + 1e-7)
        return norm_data, min_val, max_val

    def loss(self, y_real, y_fake, h_real, h_fake, x_real, x_tilde, x_hat, gamma):
        d_loss_real = nn.BCEWithLogitsLoss()(y_real, torch.ones_like(y_real))
        d_loss_fake = nn.BCEWithLogitsLoss()(y_fake, torch.zeros_like(y_fake))
        d_loss_fake_e = nn.BCEWithLogitsLoss()(h_real, torch.zeros_like(h_real))
        d_loss = d_loss_real + d_loss_fake + gamma * d_loss_fake_e

        g_loss_u = nn.BCEWithLogitsLoss()(y_fake, torch.ones_like(y_fake))
        g_loss_u_e = nn.BCEWithLogitsLoss()(h_fake, torch.ones_like(h_fake))
        g_loss_s = nn.MSELoss()(h_real[:, 1:, :], h_fake[:, :-1, :])

        g_loss_v1 = torch.mean(torch.abs(torch.sqrt(torch.var(x_hat, dim=0) + 1e-6) - torch.sqrt(torch.var(x_real, dim=0) + 1e-6)))
        g_loss_v2 = torch.mean(torch.abs(torch.mean(x_hat, dim=0) - torch.mean(x_real, dim=0)))
        g_loss_v = g_loss_v1 + g_loss_v2

        g_loss = g_loss_u + gamma * g_loss_u_e + 100 * torch.sqrt(g_loss_s) + 100 * g_loss_v
        e_loss_t0 = 10 * torch.sqrt(nn.MSELoss()(x_real, x_tilde))
        e_loss = e_loss_t0 + 0.1 * g_loss_s

        return d_loss, g_loss, e_loss

    def train(self):
        # Normalization
        ori_data, self.min_val, self.max_val = self.MinMaxScaler(self.ori_data)
        ori_data = torch.tensor(ori_data, dtype=torch.float).to(device)

        # TimeGAN training
        print('Start Embedding Network Training')
        for itt in range(self.iterations):
            X_mb, T_mb = batch_generator(ori_data, self.ori_time, self.batch_size)
            X_mb = torch.tensor(X_mb, dtype=torch.float).to(device)
            T_mb = torch.tensor(T_mb, dtype=torch.long).to(device)

            self.e_optimizer.zero_grad()
            self.r_optimizer.zero_grad()
            H = self.embedder(X_mb)
            X_tilde = self.recovery(H)
            e_loss_t0 = nn.MSELoss()(X_mb, X_tilde)
            e_loss_t0.backward()
            self.e_optimizer.step()
            self.r_optimizer.step()
            
            if itt+1 % 100 == 0:
                print(f'Epoch {itt}/{self.iterations} -- E_loss: {e_loss_t0.item()}')

        print('Train with Supervised Loss')
        for itt in range(self.iterations):
            self.d_optimizer.zero_grad()
            X_mb, T_mb = batch_generator(ori_data, self.ori_time, self.batch_size)
            X_mb = X_mb.clone().detach().to(device)
            T_mb = T_mb.clone().detach().to(device)

            
            H = self.embedder(X_mb)
            h_hat_sup = self.supervisor(H)
            
            S_loss = nn.MSELoss()(H[:,1:,:], h_hat_sup[:,:-1,:])

            self.s_optimizer.zero_grad()
            S_loss.backward()
            self.s_optimizer.step()
            
            if itt+1 % 100 == 0:
                print(f'Epoch {itt}/{self.iterations} -- S_loss: {S_loss.detach().item()}')

        print('Start Training Generator and Supervisor')
        for itt in range(self.iterations):
            for _ in range(2):
                X_mb, T_mb = batch_generator(ori_data, self.ori_time, self.batch_size)
                X_mb = torch.tensor(X_mb, dtype=torch.float).to(device)
                T_mb = torch.tensor(T_mb, dtype=torch.long).to(device)

                Z_mb = random_generator(self.batch_size, self.z_dim, T_mb, self.max_seq_len)
                Z_mb = torch.tensor(Z_mb, dtype=torch.float).to(device)
                
                e_hat = self.generator(Z_mb)
                h_hat = self.supervisor(e_hat)
                y_fake = self.discriminator(h_hat)
                y_fake_e = self.discriminator(e_hat)
                x_hat = self.recovery(h_hat)
                h = self.embedder(X_mb)
                h_hat_sup = self.supervisor(h)

                G_loss_S = nn.MSELoss()(h[:,1:,:], h_hat_sup[:,:-1,:])
                G_loss_U = nn.BCEWithLogitsLoss()(y_fake, torch.ones_like(y_fake))
                G_loss_Ue = nn.BCEWithLogitsLoss()(y_fake_e, torch.ones_like(y_fake_e))
                G_loss_V1 = torch.mean(torch.abs((torch.std(x_hat, [0], unbiased = False)) + 1e-6 - (torch.std(X_mb, [0]) + 1e-6)))
                G_loss_V2 = torch.mean(torch.abs((torch.mean(x_hat, [0]) - (torch.mean(X_mb, [0])))))
                G_loss_V = G_loss_V1 + G_loss_V2
                G_loss = 100*torch.sqrt(G_loss_S) + G_loss_U + G_loss_Ue + 100*G_loss_V

                self.g_optimizer.zero_grad()
                self.s_optimizer.zero_grad()
                G_loss.backward()
                self.g_optimizer.step()
                self.s_optimizer.step()
                
                h = self.embedder(X_mb)
                x_tilde = self.recovery(h)

                E_loss_t0 = nn.MSELoss()(X_mb, x_tilde)

                h_hat_sup = self.supervisor(h)

                G_loss_S = nn.MSELoss()(h[:,1:,:], h_hat_sup[:,:-1,:])

                E_loss = 10*torch.sqrt(E_loss_t0) + 0.1*G_loss_S

                self.e_optimizer.zero_grad()
                self.r_optimizer.zero_grad()
                E_loss.backward()
                self.e_optimizer.step()
                self.r_optimizer.step()

            X_mb, T_mb = batch_generator(ori_data, self.ori_time, self.batch_size)
            X_mb = torch.tensor(X_mb, dtype=torch.float).to(device)
            T_mb = torch.tensor(T_mb, dtype=torch.long).to(device)

            Z_mb = random_generator(self.batch_size, self.z_dim, T_mb, self.max_seq_len)
            Z_mb = torch.tensor(Z_mb, dtype=torch.float).to(device)

            h = self.embedder(X_mb)
            y_real = self.discriminator(h)
            e_hat = self.generator(Z_mb)
            y_fake_e = self.discriminator(e_hat)
            h_hat = self.supervisor(e_hat)
            y_fake = self.discriminator(h_hat)
            #x_hat = self.recovery(h_hat)

            self.d_optimizer.zero_grad()
            #self.g_optimizer.zero_grad()
            #self.s_optimizer.zero_grad()
            #self.r_optimizer.zero_grad()
            #self.e_optimizer.zero_grad()

            D_loss_real = nn.BCEWithLogitsLoss()
            DLR = D_loss_real(y_real, torch.ones_like(y_real))

            D_loss_fake = nn.BCEWithLogitsLoss()
            DLF = D_loss_fake(y_fake, torch.zeros_like(y_fake))

            D_loss_fake_e = nn.BCEWithLogitsLoss()
            DLF_e = D_loss_fake_e(y_fake_e, torch.zeros_like(y_fake_e))

            D_loss = DLR + DLF + DLF_e

            # check discriminator loss before updating
            check_d_loss = D_loss
            # This is the magic number 0.15 we mentioned above. Set exactly like in the original implementation
            if (check_d_loss > 0.15):
              D_loss.backward()
              self.d_optimizer.step()


            #Z_mb = random_generator(self.batch_size, self.z_dim, T_mb, self.max_seq_len)
            #Z_mb = torch.tensor(Z_mb, dtype=torch.float).to(device)

            #h = self.embedder(X_mb)
            #x_tilde = self.recovery(h)
            #e_hat = self.generator(Z_mb)
            #h_hat = self.supervisor(e_hat)
            #y_fake = self.discriminator(h_hat)
            #x_hat = self.recovery(h_hat)
            #h_hat_sup = self.supervisor(h)

            #G_loss_S = nn.MSELoss()(h[:,1:,:], h_hat_sup[:,:-1,:])
            #G_loss_U = nn.BCEWithLogitsLoss()(y_fake, torch.ones_like(y_fake))
            #G_loss_V1 = torch.mean(torch.abs((torch.std(x_hat, [0], unbiased = False)) + 1e-6 - (torch.std(X_mb, [0]) + 1e-6)))
            #G_loss_V2 = torch.mean(torch.abs((torch.mean(x_hat, [0]) - (torch.mean(X_mb, [0])))))
            #G_loss_V = G_loss_V1 + G_loss_V2
            #G_loss = 100 * torch.sqrt(G_loss_S) + G_loss_U + 100*G_loss_V

            #E_loss_t0 = nn.MSELoss()(X_mb, x_tilde)
            #E_loss0 = 10 * torch.sqrt(nn.MSELoss()(X_mb, x_tilde))  
            #E_loss = E_loss0  + 0.1 * G_loss_S

            #G_loss.backward(retain_graph=True)
            #E_loss.backward()

            #self.g_optimizer.step()
            #self.s_optimizer.step()
            #self.e_optimizer.step()
            #self.r_optimizer.step()

            if itt+1 % 100 == 0:
                print(f'Epoch {itt}/{self.iterations} -- G_loss: {G_loss.detach().item()} -- D_loss: {D_loss.detach().item()}')

        return self
    

    def generate(self, num_samples):
        Z_mb = random_generator(num_samples, self.z_dim, [self.max_seq_len]*num_samples, self.max_seq_len)
        Z_mb = torch.FloatTensor(Z_mb)
        Z_mb =  Z_mb.clone().detach().to(device)
        E_hat = self.generator(Z_mb)
        H_hat = self.supervisor(E_hat)
        X_hat = self.recovery(H_hat)
        generated_data = X_hat.detach()
        
        generated_data = generated_data * (self.max_val - self.min_val + 1e-7) + self.min_val
        return generated_data
    
    def get_embeddings(self, data):
        print("data",data.shape)
        self.embedder.eval()
        with torch.no_grad():
            X_mb, T_mb = batch_generator(data, self.ori_time, self.batch_size)
            X_mb = torch.tensor(X_mb, dtype=torch.float).to(device)
            H = self.embedder(X_mb)
            X_tilde = self.recovery(H)
            Z_mb = random_generator(self.batch_size, self.z_dim, T_mb, self.max_seq_len)
            Z_mb = torch.tensor(Z_mb, dtype=torch.float).to(device)
            
            X_hat = self.generator(Z_mb)
            H_hat = self.embedder(X_hat)
            embeddings = H_hat

            
        return embeddings.cpu().numpy()
    

    def train_timeganpt(self, X_train,y_train):
        parameters = {
                'hidden_dim': self.hidden_dim,
                'num_layer': self.num_layers,  # Certifique-se de que este valor seja pelo menos 1
                'iterations': self.iterations,
                'batch_size': self.batch_size,
                'module': 'lstm'
                    } 
        def extract_time(data):
            time = []
            max_seq_len = 0
            for i in range(len(data)):
                temp_time = len(data[i])
                time.append(temp_time)
                if temp_time > max_seq_len:
                    max_seq_len = temp_time
            return time, max_seq_len
        
        from models.gans.timeganpt.timegan import TimeGAN
            # Obter o formato de entrada
        reshape=True
        if(reshape):
            n_amostras=X_train.shape[0]
            X_train = X_train[:n_amostras].reshape(n_amostras, 60, 6)
        print("X_train.shape", X_train.shape)
        print("y_train.shape", y_train.shape)
        class_data = defaultdict(list)
        for X, y in zip(X_train, y_train):
            class_data[y].append(X)

        # Gerar dados sintéticos para cada classe
        synthetic_data_by_class = {}
        num_samples_per_class = 10  # Número de amostras sintéticas para gerar por classe
        embeddings_data = []

        for class_label, X_class_data in class_data.items():
            X_class_data = np.array(X_class_data)
            ori_time, _ = extract_time(X_class_data)
            
            
            # Criar e treinar o TimeGAN
            timegan = TimeGAN(X_class_data, parameters)
            #timegan.to(device)  # Mover TimeGAN para GPU
            timegan.train()  



    # Função para extrair o tempo das sequências
def extract_time(data):
        time = []
        max_seq_len = 0
        for i in range(len(data)):
            temp_time = len(data[i])
            time.append(temp_time)
            if temp_time > max_seq_len:
                max_seq_len = temp_time
        return time, max_seq_len

    # Função para gerar lotes de dados
def batch_generator(X, T, batch_size):
        no = len(X)
        idx = np.random.permutation(no)[:batch_size]
        X_mb = [torch.tensor(X[i]).to(device) for i in idx]
        T_mb = [torch.tensor(T[i]).to(device) for i in idx]
        
        return torch.stack(X_mb), torch.stack(T_mb)






