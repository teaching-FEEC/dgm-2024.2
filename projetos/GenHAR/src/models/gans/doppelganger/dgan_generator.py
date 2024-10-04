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
        #df = X_train.copy()
        #df["label"] = y_train
        reshape=True
        if reshape:
            n_amostras = X_train.shape[0]
            X_train = X_train.values.reshape(n_amostras, 60, 6)
            attributes = y_train.values.reshape(n_amostras, 1)
        
        # Configura o modelo DGAN
        self.model = DGAN(DGANConfig(
            max_sequence_len=60,
            sample_len=6,
            batch_size=self.config['parameters']['batch_size'],
            epochs=self.config['parameters']['epochs'],
        ))

        # Treina o modelo
        self.model.train_numpy(attributes=attributes, features=X_train)
        print("DGAN model training complete.")

    def generate(self, n_samples):
        if self.model is None:
            raise RuntimeError("The model has not been trained yet.")
        
        # Gera dados sintéticos
        synthetic_attributes, synthetic_features  = self.model.generate_numpy(n_samples)
        synthetic_features_flat = synthetic_features.reshape(n_samples, 60 * 6)
    
        # Criar DataFrame similar ao original X_train
        synthetic_df = pd.DataFrame(synthetic_features_flat, columns=[f'feature_{i}' for i in range(360)])
        
        # Adiciona os atributos (rótulos) como uma coluna separada
        synthetic_df['label'] = synthetic_attributes.flatten()
            
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
