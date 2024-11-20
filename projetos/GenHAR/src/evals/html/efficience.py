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
            
            # Graphs
            dcc.Graph(id='time-graph'),
            dcc.Graph(id='memory-graph'),
            dcc.Graph(id='cpu-graph'),
            dcc.Graph(id='gpu-graph')
        ])
     
    def setup_callbacks(self):
        @self.app.callback(
            [Output('time-graph', 'figure'),
             Output('memory-graph', 'figure'),
             Output('cpu-graph', 'figure'),
             Output('gpu-graph', 'figure')],
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

            # Filter the DataFrame based on selections
            filtered_df = self.df[(self.df['dataset'].isin(selected_datasets)) & (self.df['Modelo'].isin(selected_models))]

            # Execution Time bar chart
            fig_time = px.bar(
                filtered_df,
                x="dataset",
                y="Tempo de execução (s)",
                color="Modelo",
                barmode="group",
                facet_row="Fase",
                title="Tempo de Execução por Dataset e Modelo"
            )

            # Memory Usage bar chart
            fig_memory = px.bar(
                filtered_df,
                x="dataset",
                y="Memória utilizada (MB)",
                color="Modelo",
                barmode="group",
                facet_row="Fase",
                title="Memória Utilizada por Dataset e Modelo"
            )

            # CPU Usage line chart
            fig_cpu = px.bar(
                filtered_df,
                x="dataset",
                y="Uso médio de CPU (%)",
                color="Modelo",
                barmode="group",
                facet_row="Fase",
                title="Uso Médio de CPU por Dataset e Modelo"
            )

            # GPU Usage line chart, if available
            if "Memória GPU utilizada (MB)" in filtered_df.columns:
                fig_gpu = px.bar(
                    filtered_df,
                    x="dataset",
                    y="Memória GPU utilizada (MB)",
                    color="Modelo",
                    barmode="group",
                    facet_row="Fase",
                    title="Memória GPU utilizada (MB) por Dataset e Modelo"
                )
            else:
                fig_gpu = go.Figure()

            return fig_time, fig_memory, fig_cpu, fig_gpu


