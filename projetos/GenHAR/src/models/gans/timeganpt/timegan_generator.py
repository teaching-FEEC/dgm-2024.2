# from models.gans.timeganpt.modules_and_training import TimeGAN
import pickle  # Para salvar e carregar o modelo
from utils.dataset_utils import split_axis_reshape, dict_class_samples

# from models.gans.timeganpt.utils import sine_data_generation,MinMaxScaler
# from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from models.gans.timeganpt.utils import extract_time
from models.gans.timeganpt.utils import random_generator
from models.gans.timeganpt.utils import MinMaxScaler
from models.gans.timeganpt.utils import MinMaxUnscaler
from models.gans.timeganpt.utils import sine_data_generation
from models.gans.timeganpt.utils import visualization

from models.gans.timeganpt.modules_and_training import TimeGAN
from models.gans.timeganpt.modules_and_training import Time_GAN_module
import pandas as pd


class TimeGANGenerator:
    def __init__(self, m_config):
        self.m_config = m_config
        self.models = {}  # Dicionário para armazenar modelos TimeGAN por classe
        parameters = dict()
        parameters["module"] = "gru"
        parameters["hidden_dim"] = 24
        parameters["num_layers"] = 3
        parameters["iterations"] = 100
        parameters["batch_size"] = 128
        parameters["epoch"] = 500
        self.parameters = self.m_config["parameters"]

    def reshape_data(self, x_df, y_train):
        x_data = split_axis_reshape(x_df)
        class_data_dict = dict_class_samples(x_data, y_train.copy())
        for class_label in class_data_dict.keys():
            try:
                train_data = class_data_dict[class_label]

                transform_data = train_data.transpose(0, 2, 1).reshape(-1, 6)
                # scaler = StandardScaler()
                # scaler.fit(transform_data)
                # transform_data = scaler.transform(transform_data)
                transform_data = MinMaxScaler(transform_data)
                train_data = transform_data.reshape(train_data.shape[0], 60, 6).transpose(0, 2, 1)
            except:
                print("erro train tome gan")
        return train_data

    def gen_data_timegan(self, data, Generator, Recovery):
        no, seq_len, dim = data.shape[0], data.shape[1], data.shape[2]
        random_test = random_generator(no, dim, extract_time(data)[0], extract_time(data)[1])
        test_sample = Generator(
            torch.tensor(
                random_generator(no, dim, extract_time(data)[0], extract_time(data)[1])
            ).float()
        )[0]
        test_sample = torch.reshape(test_sample, (no, seq_len, self.parameters["hidden_dim"]))
        test_recovery = Recovery(test_sample)
        test_recovery = torch.reshape(test_recovery[0], (no, seq_len, dim)).detach()
        # visualization(data, test_recovery.detach().numpy(), 'tsne')
        # visualization(data, test_recovery.detach().numpy(), 'pca')
        return test_recovery

    def train(self, X_train, y_train):
        # output_size = 20
        gamma = 1
        # no, seq_len, dim = 12800, 24, 5

        self.Generator = {}
        self.Embedder = {}
        self.Supervisor = {}
        self.Recovery = {}
        self.Discriminator = {}
        self.checkpoints = {}
        self.data = {}
        self.scaler = {}

        x_df = X_train.copy()
        self.columns_names = x_df.columns

        x_data = split_axis_reshape(x_df)
        class_data_dict = dict_class_samples(x_data, y_train.copy())
        for class_label in class_data_dict.keys():
            train_data = class_data_dict[class_label]
            transform_data = train_data.transpose(0, 2, 1).reshape(-1, 6)
            # scaler = StandardScaler()
            # scaler.fit(transform_data)
            # transform_data = scaler.transform(transform_data)
            transform_data, min, max = MinMaxScaler(transform_data)
            self.scaler[class_label] = {"min": min, "max": max}
            data = transform_data.reshape(train_data.shape[0], 60, 6)  # .transpose(0, 2, 1)

            # data= self.reshape_data(x_df,y_train)
            # train_data = torch.Tensor(train_data[:(train_data.shape[0] // 128) * 128])
            # data = sine_data_generation(no, seq_len, dim)

            # data = torch.Tensor(data)

            # data=self.reshape_data(X_train.copy(),y_train)
            # print(data.shape)
            # data=X_train.copy().values.reshape(-1, 360,1)
            # data = MinMaxScaler(data)
            data = torch.Tensor(data[: (data.shape[0] // 128) * 128])
            print(data.shape)

            Generator, Embedder, Supervisor, Recovery, Discriminator, checkpoints = TimeGAN(
                data, self.parameters
            )
            self.Generator[class_label] = Generator
            self.Embedder[class_label] = Embedder
            self.Supervisor[class_label] = Supervisor
            self.Recovery[class_label] = Recovery
            self.Discriminator[class_label] = Discriminator
            self.checkpoints[class_label] = checkpoints
            self.data[class_label] = data

    def generate(self, num_samples_per_class):
        synthetic_df = pd.DataFrame()

        # Loop para gerar dados sintéticos por classe
        for class_label in self.Generator.keys():
            # Obter o modelo do gerador correspondente
            generator = self.Generator[class_label]
            recovery = self.Recovery[class_label]
            data = self.data[class_label]
            synthetic_data = self.gen_data_timegan(data, generator, recovery)
            synthetic_data = synthetic_data.reshape(-1, 6)
            synthetic_data = MinMaxUnscaler(synthetic_data, self.scaler[class_label])
            print(self.scaler[class_label])
            print(synthetic_data.shape)
            synthetic_data = synthetic_data.reshape(-1, 60, 6)
            print(synthetic_data.shape)
            synthetic_data = synthetic_data.transpose(2, 1)
            synthetic_data = synthetic_data.reshape(synthetic_data.shape[0], -1).detach().numpy()
            # if isinstance(synthetic_data, torch.Tensor):
            #    synthetic_data = synthetic_data.cpu().numpy()

            # Criar um DataFrame para os dados sintéticos da classe atual
            class_df = pd.DataFrame(synthetic_data)
            class_df.columns = self.columns_names
            # Adicionar uma coluna para a classe
            class_df["label"] = [class_label] * synthetic_data.shape[0]

            # Concatenar com o DataFrame geral
            synthetic_df = pd.concat([synthetic_df, class_df], ignore_index=True)

        return synthetic_df

    def save_model(self, file_path):
        # Salvar o dicionário de modelos treinados como um arquivo .pkl
        with open(file_path, "wb") as f:
            pickle.dump(self.models, f)

    def load_model(self, file_path):
        # Carregar o dicionário de modelos a partir de um arquivo .pkl
        with open(file_path, "rb") as f:
            self.models = pickle.load(f)
