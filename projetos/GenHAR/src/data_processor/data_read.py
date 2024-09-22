from data_processor.standartized_balanced import StandardizedViewDataset

# Leitura dos datasets
class DataRead:
    def __init__(self, dataset_config):
        self.df_train = None
        self.df_test = None
        self.df_val = None

        if dataset_config['type'] == 'standartized_balanced':
            try:
                # Carregar o dataset de treino
                svd_train = StandardizedViewDataset(data_folder=dataset_config["path"], type='train')
                df_train = svd_train.load_dataset(dataset_config["name"], dataset_config['sensors'])
                self.y_train = df_train['standard activity code']
                self.x_train = df_train.drop(columns=['standard activity code'])

                # Carregar o dataset de teste
                svd_test = StandardizedViewDataset(data_folder=dataset_config["path"], type='test')
                df_test = svd_test.load_dataset(dataset_config["name"], dataset_config['sensors'])
                self.y_test = df_test['standard activity code']
                self.x_test = df_test.drop(columns=['standard activity code'])

                # Carregar o dataset de validação
                svd_val = StandardizedViewDataset(data_folder=dataset_config["path"], type='validation')
                df_val = svd_val.load_dataset(dataset_config["name"], dataset_config['sensors'])
                self.y_val = df_val['standard activity code']
                self.x_val = df_val.drop(columns=['standard activity code'])

            except Exception as e:                 
                print(f"Error in standardized_balanced dataset read: {e}")

    def get_dataframes(self):
        return self.x_train,self.y_train,self.x_test,self.y_test,self.x_val,self.y_val
    
    def get_report(self,config):
        return ""
    def get_plots(self,config):
        return ""
    def get_metrics(self,config):
        return ""
