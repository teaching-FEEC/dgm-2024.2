import pandas as pd
import os

class StandardizedViewDataset:
    def __init__(self, data_folder,type='train'):
        """Inicializa a classe com o caminho da pasta principal onde os datasets estão armazenados."""
        self.data_folder = data_folder
        self.label_column = "standard activity code"
        self.standard_activity_code_names = {
            0: "sit",
            1: "stand",
            2: "walk",
            3: "stair up",
            4: "stair down",
            5: "run",
            6: "stair up and down",
        }
        self.type=type
        self.df = pd.DataFrame()  # Inicializa um DataFrame vazio



    def load_dataset(self, dataset_name, sensors):
        """Carrega o dataset e seleciona as colunas com base nos sensores especificados."""
        df = pd.read_csv(os.path.join(self.data_folder, dataset_name +"/" +self.type+'.csv'))
        
        # Inicializa a lista de colunas selecionadas
        selected_columns = []
        
        # Adiciona as colunas de aceleração, se especificado
        if 'accel' in sensors:
            accel_columns_x = [col for col in df.columns if col.startswith('accel-x')]
            accel_columns_y = [col for col in df.columns if col.startswith('accel-y')]
            accel_columns_z = [col for col in df.columns if col.startswith('accel-z')]
            #accel_columns = [col for col in accel_columns if col in ['accel-x', 'accel-y', 'accel-z']]
            accel_columns = accel_columns_x+accel_columns_y+ accel_columns_z
            selected_columns += accel_columns

        
        # Adiciona as colunas de giroscópio, se especificado
        if 'gyro' in sensors:
            gyro_columns_x = [col for col in df.columns if col.startswith('gyro-x')]
            gyro_columns_y = [col for col in df.columns if col.startswith('gyro-y')]
            gyro_columns_z = [col for col in df.columns if col.startswith('gyro-z')]
        #gyro_columns = [col for col in self.df.columns if col.startswith('gyro-')]
            selected_columns += gyro_columns_x+gyro_columns_y+gyro_columns_z
        
        # Adiciona a coluna de labels
        selected_columns.append('standard activity code')
        
        # Retorna o DataFrame filtrado com as colunas selecionadas
        return df[selected_columns]

    def combine_datasets(self, dataset_names, sensors):
        """Combina múltiplos datasets em um único DataFrame."""
        combined_df = pd.DataFrame()
        
        for dataset_name in dataset_names:
            df = self.load_dataset(dataset_name, sensors)
            # Adiciona uma coluna para identificar a origem do dataset
            df['source'] = dataset_name
            # Concatena os DataFrames
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        return combined_df


    def get_accel_x(self):
        """Retorna as colunas do acelerômetro eixo X."""
        return self.df[[col for col in self.df.columns if col.startswith('accel-x')]]

    def get_accel_y(self):
        """Retorna as colunas do acelerômetro eixo Y."""
        return self.df[[col for col in self.df.columns if col.startswith('accel-y')]]

    def get_accel_z(self):
        """Retorna as colunas do acelerômetro eixo Z."""
        return self.df[[col for col in self.df.columns if col.startswith('accel-z')]]

    def get_gyro_x(self):
        """Retorna as colunas do giroscópio eixo X."""
        return self.df[[col for col in self.df.columns if col.startswith('gyro-x')]]

    def get_gyro_y(self):
        """Retorna as colunas do giroscópio eixo Y."""
        return self.df[[col for col in self.df.columns if col.startswith('gyro-y')]]

    def get_gyro_z(self):
        """Retorna as colunas do giroscópio eixo Z."""
        return self.df[[col for col in self.df.columns if col.startswith('gyro-z')]]

    def get_labels(self):
        """Retorna a coluna de labels de atividade."""
        return self.df[self.label_column]

    def get_accel_gyro_labels(self):
        """Retorna as colunas do acelerômetro, giroscópio e labels."""
        accel_columns = [col for col in self.df.columns if col.startswith('accel-')]
        gyro_columns = [col for col in self.df.columns if col.startswith('gyro-')]
        return self.df[accel_columns + gyro_columns + [self.label_column]]



