import os
import pandas as pd
from typing import List
from utils import log
class Metrics:
    def __init__(self, dataset_folder,result_folder):        
        self.dataset_folder = dataset_folder
        self.folder_reports_csv=result_folder
        self.datasets_name = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]

    def read_synth_df(self,folder_df,synth_file_name):
        pd.read_csv(f"{folder_df}/{synth_file_name}")

    def read_datasets(self,folder_df):
        df_train = pd.read_csv(f"{self.dataset_folder}/{folder_df}/train.csv")
        df_test = pd.read_csv(f"{self.dataset_folder}/{folder_df}/test.csv")
        df_val = pd.read_csv(f"{self.dataset_folder}/{folder_df}/val.csv")
        return df_train,df_test,df_val

    def save_metrics_to_csv(self, metrics_list: List[dict], output_path: str):
        df_metrics = pd.DataFrame(metrics_list)
        df_metrics.to_csv(output_path, index=False)

    def ml_metrics(self,dataset: List[str], csvs_synth: List[str]=None,synth_comb_file_path="dataset_combination_None"):        
        from evals.metrics.utility_ml import ModelEvaluator
        metrics_list = []  # Collect all metrics for each model and dataset
        for dataset in dataset:
            log.print_debug(f"eval {dataset}")
            df_train,df_test,df_val=self.read_datasets(dataset)
            csvs_synth = [f for f in os.listdir(f'{self.dataset_folder}/{dataset}') if f.endswith('.csv') and 'synth' in f]      
     
            # Step 2: Define classifiers by names
            classifier_names = ["Random Forest", "SVM"]
            # Step 3: Initialize the evaluator and evaluate models
            for idx, csv_file in enumerate(csvs_synth):
                file_path = os.path.join(self.dataset_folder, dataset, csv_file)
                df_synthetic_ = pd.read_csv(file_path)
                evaluator_ml = ModelEvaluator(classifiers=classifier_names, df_train=df_train, df_test=df_test, df_val=df_val, df_synthetic=df_synthetic_, label="label", generator_name=csvs_synth[idx],                    dataset_name=dataset,  transformation_name="",
                )
                
                # Step 4: Evaluate models and collect results
                metrics = evaluator_ml.evaluate_all_classifiers_models()
                data = []
                for type_, metrics_dict in metrics.items():  # Iterar sobre el diccionario de métricas
                    for dataset_type, metrics_values in metrics_dict.items():  # Iterar sobre los tres conjuntos de datos: 'real', 'synthetic', 'mixed'
                        data = {
                                'dataset_type': dataset_type,
                                'accuracy': metrics_values.get('accuracy', None),  # Usamos .get() para evitar KeyError
                                'precision': metrics_values.get('precision', None),
                                'recall': metrics_values.get('recall', None),
                                'f1-score': metrics_values.get('f1-score', None),
                                'dataset': dataset,
                                'classifier': type_,  # Añadir el tipo de conjunto de datos
                                'generator': csvs_synth[idx].split(".")[0]  # Asegúrate de que 'csvs_synth' y 'idx' estén definidos correctamente
        }
                        metrics_list.append(data)
        

                if(synth_comb_file_path is not None):
                    dataset_c="dataset_combination_None"
                    file_path_e = os.path.join(self.dataset_folder, dataset_c, csvs_synth[idx])
                    # Verifica se o arquivo existe
                    if os.path.exists(file_path_e):
                            # Se o arquivo existir, lê o arquivo CSV
                        df_synthetic_c = pd.read_csv(file_path_e)
                        evaluator_ml_c = ModelEvaluator(
                                classifiers=classifier_names,
                                df_train=df_train,
                                df_test=df_test,
                                df_val=df_val,
                                df_synthetic=df_synthetic_c,
                                label="label",
                                generator_name=csvs_synth[idx],
                                dataset_name=dataset,
                                transformation_name="",
                            )
                        metrics_c = evaluator_ml_c.evaluate_all_classifiers_models()
                        data = []
                        for type_c, metrics_dict_c in metrics_c.items():  # Iterar sobre el diccionario de métricas
                            for dataset_type_c, metrics_values_c in metrics_dict_c.items():
                                if("real" not in dataset_type_c):
                                    data_c = {
                                    'dataset_type': f"{dataset_type_c}_comb",
                                    'accuracy': metrics_values_c.get('accuracy', None),  # Usamos .get() para evitar KeyError
                                    'precision': metrics_values_c.get('precision', None),
                                    'recall': metrics_values_c.get('recall', None),
                                    'f1-score': metrics_values_c.get('f1-score', None),
                                    'dataset': dataset,
                                    'classifier': type_c,  # Añadir el tipo de conjunto de datos
                                    'generator': csvs_synth[idx].split(".")[0]  # Asegúrate de que 'csvs_synth' y 'idx' estén definidos correctamente
                                }
                                    metrics_list.append(data_c)

        df_metrics = pd.DataFrame(metrics_list)
        output_file = f'{self.folder_reports_csv}/utlility_ml_metrics.csv'
        df_metrics.to_csv(output_file, index=False)
        print(f"Métricas salvas no arquivo {output_file}")  
           
    def calculate_feature_based_measures(self,datasets,csvs_synth=None):
        import evals.metrics.feature_based_measures as fbm
        import torch
        metrics_list=[]
        for dataset in datasets:            
            df_train,df_test,df_val=self.read_datasets(dataset)
            csvs_synth = [f for f in os.listdir(f'{self.dataset_folder}/{dataset}') if f.endswith('.csv') and 'synth' in f]      
            original_tensor = torch.tensor(df_train.iloc[:, :-1].values, dtype=torch.float32).reshape(-1, 6, 60).transpose(1,2)
            assert original_tensor.shape[1:] == (60, 6)
            for idx, csv_file in enumerate(csvs_synth):
                file_path = os.path.join(self.dataset_folder, dataset, csv_file)
                df_synthetic_ = pd.read_csv(file_path)
                
                synthetic_tensor = torch.tensor(df_synthetic_.iloc[:, :-1].values, dtype=torch.float32).reshape(-1, 6, 60).transpose(1,2)
                assert synthetic_tensor.shape[1:] == (60, 6)
                
                data = {
                                'MDD': fbm.calculate_mdd(original_tensor, synthetic_tensor),  # Usamos .get() para evitar KeyError
                                'ACD': fbm.calculate_acd(original_tensor, synthetic_tensor),
                                'SD': fbm.calculate_sd(original_tensor, synthetic_tensor),
                                'KD': fbm.calculate_kd(original_tensor, synthetic_tensor),
                                'dataset': dataset,
                                'generator': csvs_synth[idx].split(".")[0]  # Asegúrate de que 'csvs_synth' y 'idx' estén definidos correctamente
        }
                metrics_list.append(data)
        df_metrics = pd.DataFrame(metrics_list)
        output_file = f'{self.folder_reports_csv}/feature_based_measures.csv'
        df_metrics.to_csv(output_file, index=False)
        print(f"Métricas salvas no arquivo {output_file}")  
                
    def calculate_distance_based_measures(self,datasets,csvs_synth=None):
        print("calculate_distance_based_measures")
        import evals.metrics.distance_based_measures as dbm
        import torch
        metrics_list=[]
        for dataset in datasets:            
            df_train,df_test,df_val=self.read_datasets(dataset)
            csvs_synth = [f for f in os.listdir(f'{self.dataset_folder}/{dataset}') if f.endswith('.csv') and 'synth' in f]      
            original_tensor = torch.tensor(df_train.iloc[:, :-1].values, dtype=torch.float32).reshape(-1, 6, 60).transpose(1,2)
            #Randomize train data sequence
            indexes = torch.randperm(original_tensor.shape[0])
            original_tensor = original_tensor[indexes]
            assert original_tensor.shape[1:] == (60,6)
            #Calculate Real-2-Real similaruty
            split_idx = original_tensor.shape[0] // 2
            original_r2r_ed, _ = dbm.calculate_ed(original_tensor[:split_idx], original_tensor[split_idx:])
            original_r2r_dtw, _ = dbm.calculate_dtw(original_tensor[:split_idx], original_tensor[split_idx:])
            original_r2r_cos, _ = dbm.calculate_cosine(original_tensor[:split_idx], original_tensor[split_idx:])
            original_r2r_min, _ = dbm.calculate_minkowski(original_tensor[:split_idx], original_tensor[split_idx:], p=3)
            original_r2r_man, _ = dbm.calculate_manhattan(original_tensor[:split_idx], original_tensor[split_idx:])
            original_r2r_per, _ = dbm.calculate_pearson(original_tensor[:split_idx], original_tensor[split_idx:])

            for idx, csv_file in enumerate(csvs_synth):
                file_path = os.path.join(self.dataset_folder, dataset, csv_file)
                df_synthetic_ = pd.read_csv(file_path)
                print(f"eval: {csv_file}")
                
                synthetic_tensor = torch.tensor(df_synthetic_.iloc[:, :-1].values, dtype=torch.float32).reshape(-1, 6, 60).transpose(1,2)
                assert synthetic_tensor.shape[1:] == (60,6)
                ed_s2r, ed_max_s2r = dbm.calculate_ed(original_tensor, synthetic_tensor)
                dtw_s2r, dtw_max_s2r = dbm.calculate_dtw(original_tensor, synthetic_tensor)
                min_s2r, min_max_s2r = dbm.calculate_minkowski(original_tensor, synthetic_tensor, p=3)
                man_s2r, man_max_s2r = dbm.calculate_manhattan(original_tensor, synthetic_tensor)
                cos_s2r, cos_max_s2r = dbm.calculate_cosine(original_tensor, synthetic_tensor)
                per_s2r, per_max_s2r = dbm.calculate_pearson(original_tensor, synthetic_tensor)
                split_idx = synthetic_tensor.shape[0] // 2
                data = {
                    'ED-R2R': original_r2r_ed,
                    'ED-S2R': ed_s2r,
                    'ED-S2S': dbm.calculate_ed(synthetic_tensor[:split_idx], synthetic_tensor[split_idx:])[0],
                    'ED-MaxR2S': ed_max_s2r,
                    'DTW-R2R': original_r2r_dtw,
                    'DTW-S2R': dtw_s2r,
                    'DTW-S2S': dbm.calculate_dtw(synthetic_tensor[:split_idx], synthetic_tensor[split_idx:])[0],
                    'DTW-MaxR2S': dtw_max_s2r,
                    'Minkowski-R2R': original_r2r_min,
                    'Minkowski-S2R': min_s2r,
                    'Minkowski-S2S': dbm.calculate_minkowski(synthetic_tensor[:split_idx], synthetic_tensor[split_idx:], p=3)[0],
                    'Minkowski-MaxR2S': min_max_s2r,
                    'Manhattan-R2R': original_r2r_man,
                    'Manhattan-S2R': man_s2r,
                    'Manhattan-S2S': dbm.calculate_manhattan(synthetic_tensor[:split_idx], synthetic_tensor[split_idx:])[0],
                    'Manhattan-MaxR2S': man_max_s2r,
                    'Cosine-R2R': original_r2r_cos,
                    'Cosine-S2R': cos_s2r,
                    'Cosine-S2S': dbm.calculate_cosine(synthetic_tensor[:split_idx], synthetic_tensor[split_idx:])[0],
                    'Cosine-MaxR2S': cos_max_s2r,
                    'Pearson-R2R': original_r2r_per,
                    'Pearson-S2R': per_s2r,
                    'Pearson-S2S': dbm.calculate_pearson(synthetic_tensor[:split_idx], synthetic_tensor[split_idx:])[0],
                    'Pearson-MaxR2S': per_max_s2r,
                    'dataset': dataset,
                    'generator': csvs_synth[idx].split(".")[0]  # Asegure-se de que 'csvs_synth' e 'idx' estejam definidos corretamente
                    }

                metrics_list.append(data)
        df_metrics = pd.DataFrame(metrics_list)
        output_file = f'{self.folder_reports_csv}/distance_based_measures.csv'
        df_metrics.to_csv(output_file, index=False)
        print(f"Métricas salvas no arquivo {output_file}")  
            
    def calculate_privacity_measures(self,datasets,csvs_synth=None):
        import evals.metrics.privacity_measures as pm
        import torch
        metrics_list=[]
        for dataset in datasets:            
            df_train,df_test,df_val=self.read_datasets(dataset)
            csvs_synth = [f for f in os.listdir(f'{self.dataset_folder}/{dataset}') if f.endswith('.csv') and 'synth' in f]      
            #original_tensor = torch.tensor(df_train.iloc[:, :-1].values, dtype=torch.float32).reshape(-1, 60, 6)
            for idx, csv_file in enumerate(csvs_synth):
                file_path = os.path.join(self.dataset_folder, dataset, csv_file)
                df_synthetic_ = pd.read_csv(file_path)
                
                #synthetic_tensor = torch.tensor(df_synthetic_.iloc[:, :-1].values, dtype=torch.float32).reshape(-1, 60, 6)
                
                data = {
                                'EXAC_MATCH': pm.exact_match_score(df_train, df_synthetic_),  # Usamos .get() para evitar KeyError
                                'NEIG_PRIV_SCORE': pm.neighbors_privacy_score(df_train, df_synthetic_),
                                'MEM_INF_SCORE': pm.membership_inference_score(df_train, df_synthetic_),
                                'dataset': dataset,
                                'generator': csvs_synth[idx].split(".")[0]  # Asegúrate de que 'csvs_synth' y 'idx' estén definidos correctamente
        }
                metrics_list.append(data)
        df_metrics = pd.DataFrame(metrics_list)
        output_file = f'{self.folder_reports_csv}/privacity_measures.csv'
        df_metrics.to_csv(output_file, index=False)
        print(f"Métricas salvas no arquivo {output_file}")  
                


        


