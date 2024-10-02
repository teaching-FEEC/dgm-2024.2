import sys
from models.vae.vrae import VRAE
import models.gans.doppelganger.dgan as dpgan
import numpy as np
import torch
from collections import defaultdict
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


class DataGenerate:
    def __init__(self, m_config):
        self.m_config = m_config

    def timeganpt(self, X_train, y_train):
        parameters = {
            "hidden_dim": 6,
            "num_layer": 2,  # Certifique-se de que este valor seja pelo menos 1
            "iterations": 5000,
            "batch_size": 64,
            "module": "lstm",
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
        reshape = True
        if reshape:
            n_amostras = X_train.shape[0]
            X_train = X_train[:n_amostras].reshape(n_amostras, 60, 6)
        print("X_train.shape", X_train.shape)
        print("y_train.shape", y_train.shape)
        class_data = defaultdict(list)
        for X, y in zip(X_train, y_train):
            class_data[y].append(X)

        # Gerar dados sintéticos para cada classe
        synthetic_data_by_class = {}
        num_samples_per_class = 10  # Número de amostras sintéticas fara gerar por classe
        embeddings_data = []

        for class_label, X_class_data in class_data.items():
            X_class_data = np.array(X_class_data)
            ori_time, _ = extract_time(X_class_data)

            # Criar e treinar o TimeGAN
            timegan = TimeGAN(X_class_data, parameters)
            # timegan.to(device)  # Mover TimeGAN para GPU
            timegan.train()

    def train(self, X_train, y_train):
        if self.m_config["name"] == "vrae":

            print("vae")

        elif self.m_config["name"] == "timegantf":
            from models.gans.timegantf.timegan import timegan

            no, dim = 10000, 5
            seq_len = 24
            ori_data = self.sine_data_generation(no, seq_len, dim)
            parameters = self.m_config["parameters"]
            generated_data = timegan(ori_data, parameters)
            print("Finish Synthetic Data Generation")

        elif self.m_config["name"] == "timeganpt":
            self.timeganpt(X_train, y_train)

        elif self.m_config["name"] == "Doppelgangerger":
            synthetic_df = dpgan.dgan(self.m_config, X_train, y_train)

        elif self.m_config["name"] == "diffusion_unet1d":
            from models.diffusion.diffusion import DiffusionTrainer
            from models.diffusion.unet.unet_1d import UNet

            params = self.m_config["parameters"]
            model = UNet(in_channel=params["in_channel"],
                         out_channel=params["out_channel"],
                         seq_length=X_train.shape[-1]).to(device)
            diffusion = DiffusionTrainer(model, X_train.shape[-1], X_train.shape[1], epochs=self.m_config['epochs'])
            diffusion.train(X_train)
            synthetic_samples = diffusion.sample(batch_size=self.m_config['n_gen_samples'])
            synthetic_samples = synthetic_samples.view(synthetic_samples.shape[0], -1)
            synthetic_df = pd.DataFrame(synthetic_samples.numpy())
            synthetic_df['label'] = [-1] * self.m_config['n_gen_samples']

        return synthetic_df
