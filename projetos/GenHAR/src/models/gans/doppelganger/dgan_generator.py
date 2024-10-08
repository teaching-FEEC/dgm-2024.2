import numpy as np
import pandas as pd
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType
import os
import torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # ou ":16:8"

import numpy as np
import torch

class DCGANGenerator:
    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self, X_train, y_train):
        # Adiciona a coluna de labels ao dataframe
        df = X_train.copy()
        self.columns_names = df.columns

        # Split axis
        num_samples = df.shape[0]
        train_data = df.values.reshape(num_samples, 6, -1)
        self.seq_length = train_data.shape[-1]

        # Reshape to (seq_length * num_samples, num_axis)
        train_data = train_data.transpose(0, 2, 1)
        attributes = y_train.values.reshape(num_samples, 1)

        # Configura o modelo DGAN
        self.model = DGAN(
            DGANConfig(
                attribute_noise_dim=2,
                feature_noise_dim=2,
                attribute_num_units=10,
                feature_num_layers=2,
                feature_num_units=24,
                use_attribute_discriminator=False,
                discriminator_learning_rate=3e-4,
                generator_learning_rate=3e-4,
                max_sequence_len=self.seq_length,
                apply_feature_scaling=True,
                apply_example_scaling=False,
                sample_len=self.config["parameters"]["sample_len"],
                batch_size=self.config["parameters"]["batch_size"],
                generator_learning_rate=1e-4,
                discriminator_learning_rate=1e-4,
                epochs=self.config["parameters"]["epochs"],
            )
        )

        # Treina o modelo
        history=self.model.train_numpy(
            attributes=attributes, features=train_data, attribute_types=[OutputType.DISCRETE]
        )
        print("DGAN model training complete.")

    def generate(self, n_samples):
        if self.model is None:
            raise RuntimeError("The model has not been trained yet.")

        # Definir semente diferente para a geração
        torch.manual_seed(np.random.randint(0, 10000))

        # Gera dados sintéticos
        synthetic_attributes, synthetic_features = self.model.generate_numpy(n_samples)
        synthetic_features_flat = (
            np.array(synthetic_features).transpose(0, 2, 1).reshape(n_samples, 60 * 6)
        )

        # Criar DataFrame similar ao original X_train
        synthetic_df = pd.DataFrame(
            synthetic_features_flat, columns=self.columns_names
        )

        # Adiciona os atributos (rótulos) como uma coluna separada
        synthetic_df["label"] = synthetic_attributes.flatten()

        print(f"{n_samples} synthetic samples generated.")
        return synthetic_df

    def save_model(self, path):
        if self.model is None:
            raise RuntimeError("No model to save. Please train the model first.")

        # Salvar o estado do modelo DGAN
        self.model.save(path)
        print(f"Model saved at {path}")

    def load_model(self, path):
        # Carregar o modelo salvo
        self.model = DGAN.load(path)
        print(f"Model loaded from {path}")
