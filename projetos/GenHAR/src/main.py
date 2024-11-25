import sys,os,yaml,argparse,subprocess
from models.data_generate import DataGenerate
from data_processor.data_transform import Transform
from data_processor.data_read import DataRead
from utils import log, setup
from evals.metrics.efficience import EfficiencyMonitor
import matplotlib.pyplot as plt

setup.set_seeds()
REPO_ROOT_DIR = "../"
sys.path.append(os.path.dirname(REPO_ROOT_DIR))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_dataset(folder,filename,x_t,y=None):
    df_t = x_t.copy()
    if(y is not None):
        df_t["label"] = y
    try:
        os.makedirs(folder, exist_ok=True)
        if df_t is not None:
            file_path = f"{folder}/{filename}"
            df_t.to_csv(file_path, index=False)
            log.print_debug(f"Data saved to {file_path}")
        else:
            log.warning("No  data to save.")
    except Exception as e:
            log.print_err(f"Error saving  data: {e}")


def generate(config):    
    for dataset_conf in config["datasets"]:
        data_reader = DataRead(dataset_conf)
        x_train, y_train, x_test, y_test, x_val, y_val = data_reader.get_dataframes()
        log.print_debug(f"dataset info-> name:{dataset_conf['name']}, shape:{x_train.shape} ")

        for transformation_config in config["transformations"]:
            transform = Transform(transformation_config, x_train, x_test, x_val)
            x_t_train, x_t_test, x_t_val = transform.get_data_transform()
            log.print_debug(f"{dataset_conf['name']}{x_t_train.shape} transform info-> name:{transformation_config['name']}, shape:{x_t_train.shape} ")

            folder_save = f"{config['folder_save_data']}/{dataset_conf['name']}_{transformation_config['name']}"
            save_dataset(folder_save, "train.csv", x_t_train, y_train)
            save_dataset(folder_save, "test.csv", x_t_test, y_test)
            save_dataset(folder_save, "val.csv", x_t_val, y_val)

            for generative_model_config in config["generative_models"]:
                log.print_debug(f"{dataset_conf['name']}{x_t_train.shape} transform info-> name:{transformation_config['name']}, shape:{x_t_train.shape} ")

                data_generate = DataGenerate(generative_model_config, dataset_conf["name"], transformation_config["name"])

                # Monitorar e salvar métricas do treinamento
                monitor = EfficiencyMonitor()
                monitor.start()
                data_generate.train(x_t_train, y_train)
                training_metrics = monitor.stop()
                
                # Salva métricas de treinamento
                training_metrics["Modelo"] = generative_model_config["name"]
                training_metrics["Fase"] = "Treinamento"
                training_metrics["dataset"] = f"{dataset_conf['name']}_{transformation_config['name']}"
                monitor.save_metrics(training_metrics, f"{config['folder_reports']}/metrics_csv/efficience_metrics.csv", format="csv")

                # Monitorar e salvar métricas da geração de dados
                monitor.start()
                synthetic_df = data_generate.generate()
                generation_metrics = monitor.stop()

                # Salva métricas da geração de dados
                generation_metrics["Modelo"] = generative_model_config["name"]
                generation_metrics["Fase"] = "Geração"
                generation_metrics["dataset"] = f"{dataset_conf['name']}_{transformation_config['name']}"
                monitor.save_metrics(generation_metrics, f"{config['folder_reports']}/metrics_csv/efficience_metrics.csv", format="csv")

                # Salvar dados sintéticos
                save_dataset(folder_save, f"{generative_model_config['name']}_synth.csv", synthetic_df)
                loss_hist = data_generate.losses
                if len(loss_hist) > 0:
                    if type(loss_hist) is dict:
                        fig, axs = plt.subplots(len(loss_hist), 1)
                        fig.set_size_inches(10, 50)
                        for class_label, i in enumerate(loss_hist.keys()):
                            if type(loss_hist[class_label]) is dict:
                                axs[i].plot(loss_hist[class_label]["discriminator"], label="Discriminator")
                                axs[i].plot(loss_hist[class_label]["generator"], label="Generator")
                                axs[i].legend()
                            else:
                                axs[i].plot(loss_hist[class_label])
                        fig.savefig(folder_save+f"/{generative_model_config['name']}_losses.png")


                # Exibir métricas no log
                log.print_debug(f"Métricas de Treinamento - {generative_model_config['name']}: {training_metrics}")
                log.print_debug(f"Métricas de Geração de Dados - {generative_model_config['name']}: {generation_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthetic data generation and evaluation.")
    parser.add_argument("--config", type=str, help="Path to the configuration YAML file", required=True)
    parser.add_argument("--generate", action='store_true', help="train and generate samples")
    parser.add_argument("--eval", action='store_true', help="Calculate metrics and create reports")
    parser.add_argument("--visualize", action='store_true', help="Open HTML for visualization")
    
    # Carregar configuração YAML    
    args = parser.parse_args()
    config = load_config(args.config)
    results_dir = config['folder_save_data']
    eval_dir = config['folder_reports']
    
    if(args.generate):
        generate(config)
    if(args.eval):
        subprocess.run(['python', "src/main_eval.py", results_dir, eval_dir])
    if(args.visualize):
        subprocess.run(['python', "src/main_visualize.py", results_dir, eval_dir])
