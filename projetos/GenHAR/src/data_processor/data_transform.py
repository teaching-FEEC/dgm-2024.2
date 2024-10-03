import numpy as np
import pandas as pd
import torch


class Transform:

    def __init__(self, config_t, df_x_train, df_x_test, df_x_val):
        self.df_x_train = df_x_train
        self.df_x_test = df_x_test
        self.df_x_val = df_x_val

        if config_t['name']=='fft':
           self.apply_fft()
        elif config_t['name']=='umap':
            self.apply_umap()
            

        elif  config_t['name']=='None':
            self.x_t_train=self.df_x_train
            self.x_t_test=self.df_x_test
            self.x_t_val=self.df_x_val
            self.df_x_t_train = pd.DataFrame(self.x_t_train, columns=self.df_x_train.columns)
            self.df_x_t_test = pd.DataFrame(self.x_t_test, columns=self.df_x_test.columns)
            self.df_x_t_val = pd.DataFrame(self.x_t_val, columns=self.df_x_val.columns)
    
    def apply_fft(self):
        self.x_t_train=np.fft.fft(self.df_x_train.values, axis=1).real
        self.x_t_test=np.fft.fft(self.df_x_test.values, axis=1).real
        self.x_t_val=np.fft.fft(self.df_x_val.values, axis=1).real
        self.df_x_t_train = pd.DataFrame(self.x_t_train, columns=self.df_x_train.columns)
        self.df_x_t_test = pd.DataFrame(self.x_t_test, columns=self.df_x_test.columns)
        self.df_x_t_val = pd.DataFrame(self.x_t_val, columns=self.df_x_val.columns)

    def apply_umap(self):
        from umap import UMAP
        umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=6, random_state=42)
        self.x_t_train = umap_model.fit_transform(self.df_x_train)
        self.x_t_test=umap_model.fit_transform(self.df_x_test) 
        self.x_t_val=umap_model.fit_transform(self.df_x_val) 
        self.df_x_t_train = pd.DataFrame(self.x_t_train)
        self.df_x_t_test = pd.DataFrame(self.x_t_test)
        self.df_x_t_val = pd.DataFrame(self.x_t_val)
        

    def get_data_transform(self):
        
        return self.df_x_t_train,self.df_x_t_test,self.df_x_t_val
    
