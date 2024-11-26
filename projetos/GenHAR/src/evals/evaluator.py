import os
from evals.metrics.metrics import Metrics
from evals.reports.reports import Reports
from utils import log
class Evaluator:
    def __init__(self, dataset_folder,result_folder):
        self.activity_names = ["sit", "stand", "walk", "stair up", "stair down", "run"]
        self.folder_metrics_csv = f"{result_folder}/metrics_csv/"  
        self.dataset_folder=dataset_folder    
        os.makedirs(self.folder_metrics_csv, exist_ok=True)        
        self.datasets_name = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
        self.metrics=Metrics(dataset_folder,self.folder_metrics_csv)
        self.reports=Reports(dataset_folder,result_folder)
                
    def calculate_metrics(self,datasets=None,models=None):
        log.print_debug("--START calculate_metrics--")
        if(datasets is not None):
            datasets_eval=datasets
        else:
            datasets_eval=self.datasets_name
        if(models is not None):
            self.ml_metrics(datasets_eval,csvs_synth=models)
        else:
           self.metrics.ml_metrics(datasets_eval) 
           self.metrics.calculate_feature_based_measures(datasets_eval)
           self.metrics.calculate_distance_based_measures(datasets_eval)
           self.metrics.calculate_privacity_measures(datasets_eval)
           
    def create_reports(self,datasets=None,models=None):
        log.print_debug("--START create_reports--")
        if(datasets is not None):
            datasets_eval=datasets
        else:
            datasets_eval=self.datasets_name
        if(models is not None):
            self.reports.eval_datasets(datasets_eval,csvs_synth=models)
        else:
            self.reports.eval_datasets(datasets_eval)
            self.reports.compare_real_synth(datasets_eval)
            self.reports.compare_synths(datasets_eval)
        
     


