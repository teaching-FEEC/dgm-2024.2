
from sklearn.metrics import accuracy_score, pairwise_distances
from scipy.stats import wasserstein_distance

##https://github.com/gretelai/public_research/blob/main/oss_doppelganger/analysis.ipynb
## https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_comparison.html
## https://itsudit.medium.com/the-jensen-shannon-divergence-a-measure-of-distance-between-probability-distributions-23b2b1146550
import plotly.io as pio
import webbrowser
import os
class Evaluator:
    def __init__(self,config,df_train,df_test,df_val,df_synthetic):
        print(config)
        self.df_train=df_train
        self.df_test=df_test
        self.df_val=df_val
        self.df_synthetic=df_synthetic

        self.config=config['evaluations']
        self.dataset=config['dataset']
        self.transform=config['transform']
        self.model=config['model']
        print(self.config)
        self.plotly_figs=[]

    def get_visualization(self):
        
        print("visual")
        tsne=True
        tsne_glob=True
        bar_count_p=False
        images_p=False
        X_train = self.df_train.drop(columns=['label']).values
        y=self.df_train['label'].values
        X_gen=self.df_synthetic.drop(columns=['label']).values
        y_gen=self.df_synthetic['label'].values
        label_names = ['sit', 'stand', 'walk', 'stair up', 'stair down', 'run']

        
        if self.config['visualizations']["scatterplot"]["general"]:
            import eval.visualization.scaterplotly as sp
            fig=sp.get_plotly_by_labels(X_train,X_gen,y,y_gen)
            self.plotly_figs.append(fig)

        if self.config['visualizations']["scatterplot"]["by_label"]:    
            fig=sp.tsne_subplots_by_labels(X_train, y, X_gen, y_gen, label_names=label_names,title=f'{self.dataset}_{self.transform}_{self.model}')
            self.plotly_figs.append(fig)

        if self.config['visualizations']["label_counts"]["only_gen"]:
            from eval.visualization import bar_count
            fig=bar_count.plotly_count_by_labels(X_gen, y_gen, class_names=label_names)
            self.plotly_figs.append(fig)
        if self.config['visualizations']["label_counts"]["shynt_real"]:  
            from eval.visualization import bar_count  
            fig= bar_count.plotly_count_by_labels_compare(X_train, y, X_gen, y_gen, class_names=None)
            self.plotly_figs.append(fig)


        if images_p:
             from eval.visualization import visualize_images
             label_name = 1  # or None to plot all samples
             #visualize_images.plot_sample_comparison(X_train, y, X_gen, y_gen, label=label_name, n_samples=1, reshape=True)
             #visualize_images.visualize_original_and_reconst_ts(X_train[0:limit], X_gen, num=10)




        if tsne_glob:
            from eval.visualization import visualization_tsne_r_s
            filename=f'reports/{self.dataset}_{self.transform}_{self.model}.pdf' 
            visualization_tsne_r_s.visualize_tsne_r_s(self.df_train,self.df_synthetic,path=filename)

        output_file=f'reports/{self.dataset}_{self.transform}{self.model}.html'
        with open(output_file, "w") as f:
            for fig in self.plotly_figs:
                f.write(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
        webbrowser.open('file://' + os.path.realpath(output_file))

    def get_metrics(self):
                print("visual")
    
                
    def get_ml(self):
          print("mch")

    
    
    def fidelity(real_data, synthetic_data):
        return wasserstein_distance(real_data.ravel(), synthetic_data.ravel())

    
    
    def utility(real_data, synthetic_data, classifier):
        classifier.fit(synthetic_data, real_data)
        predictions = classifier.predict(real_data)
        return accuracy_score(real_data, predictions)

   
   
    def diversity(synthetic_data):
        return pairwise_distances(synthetic_data).mean()
