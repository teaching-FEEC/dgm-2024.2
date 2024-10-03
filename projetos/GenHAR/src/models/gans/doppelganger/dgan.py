import numpy as np
import pandas as pd

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType, Normalization


def dgan(config, X_train, y_train):

    # Split axis
    num_samples = X_train.shape[0]
    train_data = X_train.values.reshape(num_samples, 6, -1)
    seq_length = train_data.shape[-1]
    # Reshape to (seq_length * num_samples, num_axis)
    train_data = train_data.transpose(0,2,1).reshape(-1, 6)
    train_df = pd.DataFrame(train_data)
    train_df['label'] = np.repeat(y_train.values, seq_length)
    train_df['sample'] = np.repeat(np.arange(num_samples), seq_length)

    # Train the model
    model = DGAN(
        DGANConfig(
            max_sequence_len=seq_length,
            sample_len=config["parameters"]["sample_len"],
            batch_size=config["parameters"]["batch_size"],
            epochs=config["parameters"]["epochs"],
        )
    )

    model.train_dataframe(
        train_df,
        df_style='long',
        attribute_columns=["label"],
        discrete_columns=["label"],
        example_id_column='sample',
    )

    # Generate synthetic data
    n_synth_samples = config['n_gen_samples']
    synthetic_df = model.generate_dataframe(n_synth_samples)

    synth_labels = synthetic_df['label'].values[::seq_length]
    synth_data = synthetic_df.drop(columns=['label','sample']).values
    synth_data = synth_data.reshape(n_synth_samples, 60, 6).transpose(0,2,1).reshape(n_synth_samples, -1)

    synthetic_df = pd.DataFrame(synth_data, columns=X_train.columns)
    synthetic_df['label'] = synth_labels

    return synthetic_df
