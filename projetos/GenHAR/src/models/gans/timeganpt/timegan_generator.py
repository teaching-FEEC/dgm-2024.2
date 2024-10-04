import numpy as np
import pandas as pd
import torch
from models.gans.timeganpt.timegan import TimeGAN
import pickle  # Para salvar e carregar o modelo
from utils.dataset_utils import split_axis_reshape, dict_class_samples


class TimeGANGenerator:
    def __init__(self, m_config):
        self.m_config = m_config
        self.models = {}  # Dicionário para armazenar modelos TimeGAN por classe

    def extract_time(self, data):
        time = []
        max_seq_len = 0
        for i in range(len(data)):
            temp_time = len(data[i])
            time.append(temp_time)
            if temp_time > max_seq_len:
                max_seq_len = temp_time
        return time, max_seq_len

    def train(self, X_train, y_train):
        self.columns = X_train.columns
        parameters_ = self.m_config["parameters"]
        parameters = {
            "hidden_dim": parameters_["hidden_dim"],
            "num_layer": parameters_["num_layer"],
            "iterations": parameters_["iterations"],
            "batch_size": parameters_["batch_size"],
            "module": "lstm",
        }

        # Reformatar os dados para (n_amostras, 60, 6) se necessário
        reshape = True
        if reshape:
            X_train = split_axis_reshape(X_train.values).transpose(0, 2, 1)

        class_data = dict_class_samples(X_train, y_train.values)

        # Treinar o TimeGAN para cada classe
        self.synthetic_data_by_class = {}
        # Número de amostras sintéticas a gerar por classe

        for class_label, X_class_data in class_data.items():
            X_class_data = np.array(X_class_data)
            # ori_time, _ = self.extract_time(X_class_data)

            # Criar e treinar o TimeGAN
            # Novo modelo para cada classe
            model = TimeGAN(X_class_data, parameters)
            model.train()
            # Armazenar o modelo no dicionário
            self.models[class_label] = model

    def generate(self, num_samples_per_class):
        synthetic_df = pd.DataFrame()

        for class_label, model in self.models.items():
            # Gerar dados sintéticos
            synthetic_data = model.generate(num_samples=num_samples_per_class)

            # Se os dados forem um tensor do PyTorch, movê-los para a CPU e converter para NumPy
            if isinstance(synthetic_data, torch.Tensor):
                synthetic_data = synthetic_data.cpu().numpy()

            # Reorganizar os dados de (n, 60, 6) para (n, 360)
            synthetic_data = synthetic_data.reshape(synthetic_data.shape[0], -1)

            # Criar um DataFrame para os dados sintéticos da classe atual
            class_df = pd.DataFrame(synthetic_data)
            class_df.columns=self.columns
            # Adicionar uma coluna para a classe
            class_df["label"] = class_label

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
