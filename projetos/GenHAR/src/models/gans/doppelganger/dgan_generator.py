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
<<<<<<< HEAD
        df = X_train.copy()
        self.columns_names = df.columns

        # Split axis
        num_samples = df.shape[0]
        train_data = df.values.reshape(num_samples, 6, -1)
        self.seq_length = train_data.shape[-1]
        # Reshape to (seq_length * num_samples, num_axis)
        train_data = train_data.transpose(0, 2, 1)
        attributes = y_train.values.reshape(n_amostras, 1)

        # Configura o modelo DGAN
        self.model = DGAN(
            DGANConfig(
                max_sequence_len=df.shape[1],
                sample_len=self.config["parameters"]["sample_len"],
                batch_size=self.config["parameters"]["batch_size"],
                epochs=self.config["parameters"]["epochs"],
            )
        )


        # Treina o modelo
        self.model.train_numpy(attributes=attributes, features=train_data)
        print("DGAN model training complete.")

    def generate(self, n_samples):
        if self.model is None:
            raise RuntimeError("The model has not been trained yet.")

        # Gera dados sintéticos
        #synthetic_df = self.model.generate_dataframe(n_samples)
        #print(self.model.generate_numpy(n_samples)[0].shape)
        #print(self.model.generate_numpy(n_samples)[0][0])
        #print(len(self.model.generate_numpy(n_samples)[1]))
        #print(self.model.generate_numpy(n_samples)[1][0][:,-1])

        ##print(np.unique(synthetic_df['sample'].values, return_counts=True))
        #synth_labels = synthetic_df["label"].values[:: self.seq_length]
        #synth_data = synthetic_df.drop(columns=["label", "sample"]).values
        #print(synth_data.shape)
        #synth_data = (
        #    synth_data.reshape(n_samples, 60, 6)
        #    .transpose(0, 2, 1)
        #    .reshape(n_samples, -1)
        #)

        #synthetic_df = pd.DataFrame(synth_data, columns=self.columns_names)
        #synthetic_df["label"] = synth_labels
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
