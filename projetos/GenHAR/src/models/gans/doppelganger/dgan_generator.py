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
        self.columns_names = df.X_train.columns
        df["label"] = y_train

        # Split axis
        num_samples = df.shape[0]
        train_data = df.values.reshape(num_samples, 6, -1)
        self.seq_length = train_data.shape[-1]
        # Reshape to (seq_length * num_samples, num_axis)
        train_data = train_data.transpose(0,2,1).reshape(-1, 6)
        train_df = pd.DataFrame(train_data)
        train_df['label'] = np.repeat(df['label'].values, self.seq_length)
        train_df['sample'] = np.repeat(np.arange(num_samples), self.seq_length)
        
        # Configura o modelo DGAN
        self.model = DGAN(DGANConfig(
            max_sequence_len=df.shape[1],
            sample_len=self.config['parameters']['sample_len'],
            batch_size=self.config['parameters']['batch_size'],
            epochs=self.config['parameters']['epochs']
        ))

        # Treina o modelo
        self.model.train_dataframe(
            train_df,
            df_style='long',
            attribute_columns=['label'],
            discrete_columns=['label'],
            example_id_column='sample',
        )
        print("DGAN model training complete.")

    def generate(self, n_samples):
        if self.model is None:
            raise RuntimeError("The model has not been trained yet.")
        
        # Gera dados sintéticos
        synthetic_df = self.model.generate_dataframe(n_samples)

        synth_labels = synthetic_df['label'].values[::self.seq_length]
        synth_data = synthetic_df.drop(columns=['label','sample']).values
        synth_data = synth_data.reshape(n_synth_samples, 60, 6).transpose(0,2,1).reshape(n_synth_samples, -1)

        synthetic_df = pd.DataFrame(synth_data, columns=self.columns_names)
        synthetic_df['label'] = synth_labels
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
