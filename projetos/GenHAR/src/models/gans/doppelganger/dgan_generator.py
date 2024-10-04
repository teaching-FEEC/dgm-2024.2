import numpy as np
import pandas as pd
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

class DCGANGenerator:
    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self, X_train, y_train):
        # Adiciona a coluna de labels ao dataframe
        df = X_train.copy()
        df["label"] = y_train
        
        # Configura o modelo DGAN
        self.model = DGAN(DGANConfig(
            max_sequence_len=X_train.shape[1],
            sample_len=self.config['parameters']['sample_len'],
            batch_size=self.config['parameters']['batch_size'],
            epochs=self.config['parameters']['epochs']
        ))

        # Treina o modelo
        self.model.train_dataframe(
            df,
            attribute_columns=["label"],
            discrete_columns=["label"],
        )
        print("DGAN model training complete.")

    def generate(self, n_samples):
        if self.model is None:
            raise RuntimeError("The model has not been trained yet.")
        
        # Gera dados sintéticos
        synthetic_df = self.model.generate_dataframe(n_samples)
        print(synthetic_df.columns)
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
