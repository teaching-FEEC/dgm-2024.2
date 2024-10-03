import numpy as np
import pandas as pd

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType, Normalization

def dgan(config,X_train,y_train):
   
    df=X_train
    print("size ",X_train.shape[1])
    df["label"]=y_train
    # Train the model
    model = DGAN(DGANConfig(
        max_sequence_len=X_train.shape[1]-1,
        sample_len=config['parameters']['sample_len'],
        batch_size=config['parameters']['batch_size'],
        epochs=config['parameters']['epochs'],  
        
    ))

    model.train_dataframe(
        df,
        attribute_columns=["label"],
        discrete_columns=["label"],
    )

    # Generate synthetic data
    synthetic_df = model.generate_dataframe(config['n_gen_samples'])
    return synthetic_df

    
    
