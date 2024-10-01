import sys,os
REPO_ROOT_DIR="../"
sys.path.append(os.path.dirname(REPO_ROOT_DIR))
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 


import yaml
import argparse  
from data_processor.data_read import DataRead
from data_processor.data_transform import Transform
from models.data_generate import DataGenerate
from eval.evaluator import Evaluator
# Função para carregar o arquivo de configuração YAML
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config




def main(config_path):

    # Carregar configuração YAML

    config = load_config(config_path)

    for dataset_conf in config['datasets']:
        data_reader = DataRead(dataset_conf)
        x_train,y_train,x_test,y_test,x_val,y_val= data_reader.get_dataframes()
    
        for transformation_config in config['transformations']:
            print(transformation_config)
            transform=Transform(transformation_config,x_train,x_test,x_val)
            x_t_train,x_t_test,x_t_val = transform.get_data_transform()
            
            


            #print(x_t_train.shape,y_train.shape) 
            for generative_model_config in config['generative_models']:                
                data_generate =DataGenerate(generative_model_config)
                synthetic_df= data_generate.train(x_t_train,y_train)
                df_trans_train=x_train
                df_trans_train['label']=y_train
                df_trans_test=x_test
                df_trans_test['label']=y_test
                df_trans_val=x_val
                df_trans_val['label']=y_val
                
                config_eval={}
                config_eval['evaluations']=config['evaluations']
                config_eval['dataset']=dataset_conf['name']
                config_eval['transform']=transformation_config['name']
                config_eval['model']=generative_model_config['name']
                evaluator=Evaluator(config_eval,df_trans_train,df_trans_test,df_trans_val,synthetic_df)
                evaluator.get_visualization()
                evaluator.get_metrics()
                evaluator.get_ml()




              
         
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run synthetic data generation and evaluation.')
    parser.add_argument('--config', type=str, help='Path to the configuration YAML file', required=True)
    args = parser.parse_args()

    # Executa o processo principal com o caminho do YAML passado pela linha de comando
    main(args.config)
