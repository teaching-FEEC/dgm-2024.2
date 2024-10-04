import argparse
import sys
from eval.evaluator import Evaluator
from models.data_generate import DataGenerate
from data_processor.data_transform import Transform
from data_processor.data_read import DataRead
import yaml
from utils import log, setup
import time
import os

# Define as seeds para repredutibilidade
setup.set_seeds()

# https://github.com/YihaoAng/TSGBench/tree/main?tab=readme-ov-file
REPO_ROOT_DIR = "../"
sys.path.append(os.path.dirname(REPO_ROOT_DIR))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Função para carregar o arquivo de configuração YAML
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(config_path):

    # Carregar configuração YAML
    config = load_config(config_path)

    for dataset_conf in config["datasets"]:
        data_reader = DataRead(dataset_conf)
<<<<<<< HEAD
        x_train, y_train, x_test, y_test, x_val, y_val = data_reader.get_dataframes()
        log.print_debug(f"dataset info-> name:{dataset_conf['name']}, shape:{x_train.shape} ")

        for transformation_config in config["transformations"]:
            transform = Transform(transformation_config, x_train, x_test, x_val)
            x_t_train, x_t_test, x_t_val = transform.get_data_transform()
            log.print_debug(
                f"transform info-> name:{transformation_config['name']}, shape:{x_t_train.shape} "
            )

            for generative_model_config in config["generative_models"]:
                data_generate = DataGenerate(generative_model_config)
                start_time = time.time()
                data_generate.train(x_t_train, y_train)
                training_time = time.time() - start_time
                synthetic_df = data_generate.generate()

                # print(synthetic_df.head())
                break

                df_trans_train = x_t_train
                df_trans_train["label"] = y_train
                df_trans_test = x_t_test
                df_trans_test["label"] = y_test
                df_trans_val = x_t_val
                df_trans_val["label"] = y_val

                config_eval = {}
                config_eval["evaluations"] = config["evaluations"]
                config_eval["dataset"] = dataset_conf["name"]
                config_eval["transform"] = transformation_config["name"]
                config_eval["model"] = generative_model_config["name"]
                config_eval["training_time"] = training_time

                evaluator = Evaluator(
                    config_eval, df_trans_train, df_trans_test, df_trans_val, synthetic_df
                )
=======
        x_train,y_train,x_test,y_test,x_val,y_val= data_reader.get_dataframes()
        log.print_debug(f"dataset info-> name:{dataset_conf['name']}, shape:{x_train.shape} ")       
        
        
    
        for transformation_config in config['transformations']:            
            transform=Transform(transformation_config,x_train,x_test,x_val)
            x_t_train,x_t_test,x_t_val = transform.get_data_transform()
            log.print_debug(f"{dataset_conf['name']}{x_t_train.shape} transform info-> name:{transformation_config['name']}, shape:{x_t_train.shape} ")

            for generative_model_config in config['generative_models']:   
                log.print_debug(f"{dataset_conf['name']}{x_t_train.shape} transform info-> name:{transformation_config['name']}, shape:{x_t_train.shape} ")
        
                data_generate =DataGenerate(generative_model_config,dataset_conf['name'],transformation_config['name'])
                start_time = time.time()
                data_generate.train(x_t_train,y_train)
                training_time = time.time() - start_time                
                synthetic_df=data_generate.generate()
                
                #print(synthetic_df.head())
                
                
                
                df_trans_train_=x_t_train.copy()
                df_trans_train_['label']=y_train
                df_trans_test_=x_t_test.copy()
                df_trans_test_['label']=y_test
                df_trans_val_=x_t_val.copy()
                df_trans_val_['label']=y_val
                
                config_eval={}
                config_eval['evaluations']=config['evaluations']
                config_eval['dataset']=dataset_conf['name']
                config_eval['transform']=transformation_config['name']
                config_eval['model']=generative_model_config['name']
                config_eval['training_time']=training_time
                log.print_debug(f"----Eval-----")

                evaluator=Evaluator(config_eval,df_trans_train_,df_trans_test_,df_trans_val_,synthetic_df)
>>>>>>> first_version_framew
                evaluator.eval()
                # evaluator.get_metrics()
                # evaluator.get_ml()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthetic data generation and evaluation.")
    parser.add_argument(
        "--config", type=str, help="Path to the configuration YAML file", required=True
    )
    args = parser.parse_args()

    # Executa o processo principal com o caminho do YAML passado pela linha de comando
    main(args.config)
