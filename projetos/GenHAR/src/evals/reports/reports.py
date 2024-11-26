import os
import pandas as pd
from utils import log
from evals.reports.ts_dataset_eval import TSDatasetEval
from evals.reports.real_synth_comp import RealSyntheticComparator
from evals.reports.synth_compare import SyntheticComparator
import utils.report_ut as r_ut
class Reports:
    def __init__(self, dataset_folder,result_folder):        
        self.dataset_folder = dataset_folder
        self.folder_images = f"{result_folder}/images/"  
        self.folder_reports_pdf = f"{result_folder}/reports_pdf/"
        os.makedirs(self.folder_reports_pdf, exist_ok=True)
        os.makedirs(self.folder_images, exist_ok=True)
        self.datasets_name = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
        self.activity_names = ["sit", "stand", "walk", "stair up", "stair down", "run"]
        self.label_col="label"

    def read_datasets(self,folder_df):
        df_train = pd.read_csv(f"{self.dataset_folder}/{folder_df}/train.csv")
        df_test = pd.read_csv(f"{self.dataset_folder}/{folder_df}/test.csv")
        df_val = pd.read_csv(f"{self.dataset_folder}/{folder_df}/val.csv")
        return df_train,df_test,df_val

    def eval_dataset(self, dataset, dataset_name,type_dataset):
        try:
            evaluator = TSDatasetEval(
                dataset,
                label_col="label",
                label_names=self.activity_names,
            )
            log.print_debug(f"....iando reportes para o dataset...{dataset_name} {type_dataset}")

            fig1 = evaluator.plot_samples()
            fig2 = evaluator.tsne_plot()
            fig3 = evaluator.plot_random_sensor_samples_single_dataset(
                dataset, self.activity_names, num_samples=6, sensor="gyro"
            )
            fig4 = evaluator.plot_random_sensor_samples_single_dataset(
                dataset, self.activity_names, num_samples=6, sensor="accel"
            )
            fig5 = evaluator.plot_i_samples(n_samples=10, reshape=True)
            # fig6=evaluator.plot_autocorrelation()
            # fig10 = evaluator.plot_pca_correlation()
            # fig3=evaluator.histogram_density()
            # fig4=evaluator.plot_acf_pacf_all()

            r_ut.save_fig_pdf(f"{self.folder_reports_pdf}/{dataset_name}/{type_dataset}.pdf", fig1, fig2, fig3, fig4, fig5)
        except Exception as e:
            log.print_err(f"REPORTS PDF {dataset_name} {type_dataset} An error occurred: {e}")

    def eval_datasets(self,datasets_eval):              
        for dataset in datasets_eval:
            os.makedirs(f'{self.folder_reports_pdf}/{dataset}', exist_ok=True)
            df_train,df_test,df_val=self.read_datasets(dataset)  
            self.eval_dataset(df_train,dataset, "original")  
            csvs_synth = [f for f in os.listdir(f'{self.dataset_folder}/{dataset}') if f.endswith('.csv') and 'synth' in f]    
            for idx, csv_file in enumerate(csvs_synth):
                    file_path = os.path.join(self.dataset_folder, dataset, csv_file)
                    df_synthetic= pd.read_csv(file_path)
                    self.eval_dataset(df_synthetic,dataset, csv_file.split('.')[0])

    def compare_real_synth_plots(self,df_original,df_synthetic,dataset_name,generator_name):
        comparator = RealSyntheticComparator(df_original,df_synthetic, self.label_col, self.activity_names)
        fig0=comparator.visualize_distribution("", "disti")
        fig0.savefig(f"{self.folder_images}{dataset_name}/_{generator_name}_distribution.jpg", format='jpg', dpi=300)
                        # Comparar distribuição de classes
        #fig1 = comparator.compare_class_distribution()

                        # Comparar t-SNE
        fig2 = comparator.compare_tsne()
        fig3 = comparator.visualize_tsne_unlabeled()
        fig3.savefig(f"{self.folder_images}/{dataset_name}/_{generator_name}_tsne.jpg", format='jpg', dpi=300)

        fig4 = comparator.tsne_subplots_by_labels()
        fig5 = comparator.plot_samplesT_by_label(num_samples=3)

                        # Comparar amostras aleatórias
        fig6 = comparator.compare_images(reshape=True)
                        # fig4=comparator.visualize_distribution()
                        # Comparar matrizes de correlação

        fig7 = comparator.plot_random_sensor_samples_comparison()
        fig8 = comparator.plot_random_samples_comparison_by_labels(num_samples=5)
        r_ut.save_fig_pdf(f"{self.folder_reports_pdf}/{dataset_name}/_{generator_name}_real_vs_gen.pdf",
                            fig0,
                            #fig1,
                            fig2,
                            fig3,
                            fig4,
                            fig5,
                            fig6,
                            fig7,
                            fig8,
                        )
        
    def compare_real_synth(self,datasets_eval):
        for dataset in datasets_eval:
            os.makedirs(f'{self.folder_reports_pdf}/{dataset}', exist_ok=True)
            os.makedirs(f'{self.folder_images}/{dataset}', exist_ok=True)
            df_train,df_test,df_val=self.read_datasets(dataset)  
            csvs_synth = [f for f in os.listdir(f'{self.dataset_folder}/{dataset}') if f.endswith('.csv') and 'synth' in f]    
            for idx, csv_file in enumerate(csvs_synth):
                    file_path = os.path.join(self.dataset_folder, dataset, csv_file)
                    df_synthetic= pd.read_csv(file_path)
                    self.compare_real_synth_plots(df_train,df_synthetic,dataset,csv_file)


    def compare_synths(self,datasets_eval):
        for dataset in datasets_eval:
            os.makedirs(f'{self.folder_reports_pdf}/{dataset}', exist_ok=True)
            os.makedirs(f'{self.folder_images}/{dataset}', exist_ok=True)
            df_train,df_test,df_val=self.read_datasets(dataset)  
            csvs_synth = [f for f in os.listdir(f'{self.dataset_folder}/{dataset}') if f.endswith('.csv') and 'synth' in f]    
            df_synths=[]
            synt_df_names=[]
            for idx, csv_file in enumerate(csvs_synth):
                    file_path = os.path.join(self.dataset_folder, dataset, csv_file)
                    df_synthetic= pd.read_csv(file_path)
                    df_synths.append(df_synthetic)
                    synt_df_names.append(csv_file)
            synths_comparetor=SyntheticComparator(df_train,df_synths,synt_df_names,activity_names=self.activity_names)
            fig_tsne_u=synths_comparetor.visualize_tsne_unlabeled()
            fig_class_u=synths_comparetor.compare_class_distribution()
            fig_dist_u=synths_comparetor.visualize_distribution()
            tsne_dist_l=synths_comparetor.tsne_subplots_by_labels()
            feat_time_l=synths_comparetor.plot_random_samples_comparison_by_labels(num_samples=5)
            feat_sens_l=synths_comparetor.plot_random_sensor_samples_comparison(num_samples=5)
            r_ut.save_fig_pdf(f"{self.folder_reports_pdf}/{dataset}/_synth_comp.pdf",
                            fig_tsne_u,fig_class_u,fig_dist_u,tsne_dist_l,feat_time_l,*feat_sens_l                     
                        )
            


