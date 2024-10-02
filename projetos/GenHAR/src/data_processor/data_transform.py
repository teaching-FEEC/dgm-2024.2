import numpy as np
import pandas as pd
import torch


class Transform:

    def __init__(self, config_t, df_x_train, df_x_test, df_x_val):
        self.df_x_train = df_x_train
        self.df_x_test = df_x_test
        self.df_x_val = df_x_val

        if config_t["name"] == "fft":
            self.apply_fft()
        elif config_t["name"] == "split":
            params = config_t["parameters"]
            if params["method"] == "sensor":
                self.x_t_train = self.split_sensors(self.df_x_train)
                self.x_t_test = self.split_sensors(self.df_x_test)
                self.x_t_val = self.split_sensors(self.df_x_val)
            elif params["method"] == "axis":
                self.x_t_train = self.split_axis(self.df_x_train)
                self.x_t_test = self.df_x_test #self.split_axis(self.df_x_test)
                self.x_t_val = self.df_x_val #self.split_axis(self.df_x_val)
            if params['convert'] == 'tensor':
                self.x_t_train = self.toTensor(self.x_t_train)
                #self.x_t_test = self.toTensor(self.x_t_test)
                #self.x_t_val = self.toTensor(self.x_t_val)
        elif config_t["name"] == "None":
            self.x_t_train = self.df_x_train
            self.x_t_test = self.df_x_test
            self.x_t_val = self.df_x_val

    def split_axis(self, df):
        axis_columns = []
        axis_columns.append(
            [col for col in df.columns if col.startswith("accel-x")])
        axis_columns.append(
            [col for col in df.columns if col.startswith("accel-y")])
        axis_columns.append(
            [col for col in df.columns if col.startswith("accel-z")])
        axis_columns.append(
            [col for col in df.columns if col.startswith("gyro-x")])
        axis_columns.append(
            [col for col in df.columns if col.startswith("gyro-y")])
        axis_columns.append(
            [col for col in df.columns if col.startswith("gyro-z")])
        return [df[columns] for columns in axis_columns]

    def split_sensors(self, df):
        sensors_columns = []
        sensors_columns.append(
            [col for col in df.columns if col.startswith("accel")])
        sensors_columns.append(
            [col for col in df.columns if col.startswith("gyro")])
        return [df[columns] for columns in sensors_columns]

    def toTensor(self, df_list):
        return torch.stack([torch.tensor(df.values) for df in df_list]).transpose(0,1)

    def apply_fft(self):
        self.x_t_train = self.to_dataframe(
            np.fft.fft(self.df_x_train.values,
                       axis=1).real, self.df_x_train.columns
        )
        self.x_t_test = self.to_dataframe(
            np.fft.fft(self.df_x_test.values,
                       axis=1).real, self.df_x_test.columns
        )
        self.x_t_val = self.to_dataframe(
            np.fft.fft(self.df_x_val.values,
                       axis=1).real, self.df_x_val.columns
        )

    def to_dataframe(self, data, cols):
        return pd.DataFrame(data, columns=cols)

    def get_data_transform(self):
        return self.x_t_train, self.x_t_test, self.x_t_val
