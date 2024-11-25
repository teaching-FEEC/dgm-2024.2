import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
class EfficiencyMetrics:
    def __init__(self, app, file):
        self.app = app
        self.df = pd.read_csv(file)
        print(self.df.columns)
        # Set up layout and callbacks
        self.app.layout = self.create_layout()
        self.setup_callbacks()

    def create_layout(self):
        return html.Div([
            html.H1("Comparação de Métricas de Eficiência por Dataset e Modelo"),
            
            # Dropdown for dataset selection
            html.Label("Selecione o Dataset:"),
            dcc.Dropdown(
                id='dataset-dropdown_eff',
                options=[{'label': dataset, 'value': dataset} for dataset in self.df['dataset'].unique()],
                value=list(self.df['dataset'].unique()),  # Adjusted for multi-selection
                multi=True
            ),
    
            # Dropdown for model selection
            html.Label("Selecione o Modelo:"),
            dcc.Dropdown(
                id='generetor-dropdown_eff',
                options=[{'label': model, 'value': model} for model in self.df['Modelo'].unique()],
                value=list(self.df['Modelo'].unique()),  # Adjusted for multi-selection
                multi=True
            ),
            
            # Graphs for Training Phase
            html.H2("Treinamento"),
            dcc.Graph(id='time-graph_train'),
            dcc.Graph(id='memory-graph_train'),
            dcc.Graph(id='cpu-graph_train'),
            dcc.Graph(id='gpu-graph_train'),

            # Graphs for Generation Phase
            html.H2("Geração"),
            dcc.Graph(id='time-graph_generate'),
            dcc.Graph(id='memory-graph_generate'),
            dcc.Graph(id='cpu-graph_generate'),
            dcc.Graph(id='gpu-graph_generate'),
        ])
     
    def setup_callbacks(self):
        @self.app.callback(
            [
                Output('time-graph_train', 'figure'),
                Output('memory-graph_train', 'figure'),
                Output('cpu-graph_train', 'figure'),
                Output('gpu-graph_train', 'figure'),
                Output('time-graph_generate', 'figure'),
                Output('memory-graph_generate', 'figure'),
                Output('cpu-graph_generate', 'figure'),
                Output('gpu-graph_generate', 'figure'),
            ],
            [Input('dataset-dropdown_eff', 'value'),
             Input('generetor-dropdown_eff', 'value')]
        )
        def update_graphs(selected_datasets, selected_models):
            # Ensure selected_datasets and selected_models are lists
            print("Callback triggered")
            if isinstance(selected_datasets, str):
                selected_datasets = [selected_datasets]
            if isinstance(selected_models, str):
                selected_models = [selected_models]
            
            # Phases
            fases_train = ['Treinamento']
            fases_generate = ['Geração']

            # Filter the DataFrame for training and generation
            filtered_df_train = self.df[(self.df['dataset'].isin(selected_datasets)) & 
                                        (self.df['Modelo'].isin(selected_models)) & 
                                        (self.df['Fase'].isin(fases_train))]

            filtered_df_generate = self.df[(self.df['dataset'].isin(selected_datasets)) & 
                                           (self.df['Modelo'].isin(selected_models)) & 
                                           (self.df['Fase'].isin(fases_generate))]

            # Training Phase Graphs
            fig_time_train = px.bar(
                filtered_df_train,
                x="dataset",
                y="Tempo de execução (s)",
                color="Modelo",
                barmode="group",
                title="Tempo de Execução no Treinamento"
            )

            fig_memory_train = px.bar(
                filtered_df_train,
                x="dataset",
                y="Memória utilizada (MB)",
                color="Modelo",
                barmode="group",
                title="Memória Utilizada no Treinamento"
            )

            fig_cpu_train = px.bar(
                filtered_df_train,
                x="dataset",
                y="Uso médio de CPU (%)",
                color="Modelo",
                barmode="group",
                title="Uso Médio de CPU no Treinamento"
            )

            fig_gpu_train = px.bar(
                filtered_df_train,
                x="dataset",
                y="Memória GPU utilizada (MB)",
                color="Modelo",
                barmode="group",
                title="Memória GPU Utilizada no Treinamento"
            ) if "Memória GPU utilizada (MB)" in filtered_df_train.columns else go.Figure()

            # Generation Phase Graphs
            fig_time_generate = px.bar(
                filtered_df_generate,
                x="dataset",
                y="Tempo de execução (s)",
                color="Modelo",
                barmode="group",
                title="Tempo de Execução na Geração"
            )

            fig_memory_generate = px.bar(
                filtered_df_generate,
                x="dataset",
                y="Memória utilizada (MB)",
                color="Modelo",
                barmode="group",
                title="Memória Utilizada na Geração"
            )

            fig_cpu_generate = px.bar(
                filtered_df_generate,
                x="dataset",
                y="Uso médio de CPU (%)",
                color="Modelo",
                barmode="group",
                title="Uso Médio de CPU na Geração"
            )

            fig_gpu_generate = px.bar(
                filtered_df_generate,
                x="dataset",
                y="Memória GPU utilizada (MB)",
                color="Modelo",
                barmode="group",
                title="Memória GPU Utilizada na Geração"
            ) if "Memória GPU utilizada (MB)" in filtered_df_generate.columns else go.Figure()

            return (
                fig_time_train, fig_memory_train, fig_cpu_train, fig_gpu_train,
                fig_time_generate, fig_memory_generate, fig_cpu_generate, fig_gpu_generate
            )
