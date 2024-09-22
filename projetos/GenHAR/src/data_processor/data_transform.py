
import numpy as np
import pandas as pd
class Transform:

    def __init__(self, config_t,df_x_train, df_x_test, df_x_val):
        self.df_x_train=df_x_train
        self.df_x_test=df_x_test
        self.df_x_val=df_x_val

        if config_t['name']=='fft':
           self.apply_fft()
        elif  config_t['name']=='None':
            self.x_t_train=self.df_x_train
            self.x_t_test=self.df_x_test
            self.x_t_val=self.df_x_val
    
    def apply_fft(self):
        self.x_t_train=np.fft.fft(self.df_x_train.values, axis=1).real
        self.x_t_test=np.fft.fft(self.df_x_test.values, axis=1).real
        self.x_t_val=np.fft.fft(self.df_x_val.values, axis=1).real

    def get_data_transform(self):
        self.df_x_t_train = pd.DataFrame(self.x_t_train, columns=self.df_x_train.columns)
        self.df_x_t_test = pd.DataFrame(self.x_t_test, columns=self.df_x_test.columns)
        self.df_x_t_val = pd.DataFrame(self.x_t_val, columns=self.df_x_val.columns)
        return self.df_x_t_train,self.df_x_t_test,self.df_x_t_val
    