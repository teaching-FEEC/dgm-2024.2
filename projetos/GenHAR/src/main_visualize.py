from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
from evals.html.utility_ml import UtilityML
from evals.html.efficience import EfficiencyMetrics
from evals.html.feature_based_measure import FeatureBasedMeasures
from evals.html.distance_based_measure import DistanceBasedMeasures
from evals.html.privacity_measure import PrivacyBasedMetrics
import argparse
class App:
    def __init__(self,results_dir, eval_dir):
        # Caminho para o arquivo de métricas
        file_ml_metrics = f"{eval_dir}/metrics_csv/utlility_ml_metrics.csv"
        file_efficence = f"{eval_dir}/metrics_csv/efficience_metrics.csv"
        file_fbm_metrics = f"{eval_dir}/metrics_csv/feature_based_measures.csv"
        file_dbm_metrics = f"{eval_dir}/metrics_csv/distance_based_measures.csv"
        file_priv_metrics= f"{eval_dir}/metrics_csv/privacity_measures.csv"
        # Inicializando a aplicação Dash
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        
        # Instanciando as classes das abas
        self.utility_ml = UtilityML(file_ml_metrics, self.app)
        self.efficiency_metrics = EfficiencyMetrics(self.app, file_efficence)  # Replace with actual CSV file path
        self.fbm_metrics = FeatureBasedMeasures(file_fbm_metrics, self.app) 
        self.dbm_metrics = DistanceBasedMeasures(file_dbm_metrics, self.app) 
        self.priv_metrics = PrivacyBasedMetrics(file_priv_metrics, self.app) 
        # Definindo o layout da aplicação com abas
        self.app.layout = html.Div([
            dcc.Tabs(id='tabs', value='utility_ml', children=[
                dcc.Tab(label='Utility ML', value='utility_ml'),
                dcc.Tab(label='feature based measures', value='feature_based_measures'),
                dcc.Tab(label='distance based measures', value='distance_based_measures'),              
                dcc.Tab(label='Eficiência', value='eficiencia'),
                dcc.Tab(label='Privacidade', value='privacity'),
                dcc.Tab(label='Distribuições Correspondentes', value='distribuicoes'),
                dcc.Tab(label='Representatividade', value='representatividade'),
                dcc.Tab(label='Cobertura', value='cobertura'),
                dcc.Tab(label='Novidade', value='novidade'),
                dcc.Tab(label='Generalização', value='generalizacao'),
                dcc.Tab(label='Semelhança de Instâncias', value='semelhanca'),
                dcc.Tab(label='Proximidade de Embedding', value='proximidade'),
                
            ]),
            html.Div(id='tab-content', children=[
                self.utility_ml.create_layout()  # Este layout será substituído por outros layouts dependendo da aba selecionada
            ])
        ])

        self.setup_callbacks()

    def setup_callbacks(self):
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('tabs', 'value')
        )
        def render_tab_content(tab):
            if tab == 'utility_ml':
                return self.utility_ml.create_layout()
            elif tab == 'eficiencia':
                print("eficc")
                self.efficiency_metrics.setup_callbacks()
                return self.efficiency_metrics.create_layout()
            elif tab == 'feature_based_measures':
                return self.fbm_metrics.create_layout()
            elif tab == 'distance_based_measures':
                return self.dbm_metrics.create_layout()
            elif tab == 'privacity':
                return self.priv_metrics.create_layout()
            
            return html.Div(f"Conteúdo para a aba: {tab}")

if __name__ == '__main__':
    #"data/results","results_eval"
    parser = argparse.ArgumentParser(description='Avaliação de resultados.')
    parser.add_argument('results_dir', type=str, help='Caminho para os resultados')
    parser.add_argument('eval_dir', type=str, help='Caminho para o diretório de avaliação')
    args = parser.parse_args()
    app = App(args.results_dir, args.eval_dir)
    app.app.run_server(debug=True, port=8051)


