
from dash import  dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html, Input, Output

class UtilityML:
    def __init__(self, file_ml_metrics, app):
        self.data = pd.read_csv(file_ml_metrics)

        self.metrics = ['accuracy', 'precision', 'recall', 'f1-score']
        
        # Opções para dropdowns
        self.dataset_options = [{'label': ds, 'value': ds} for ds in self.data['dataset'].unique()]
        self.classifier_options = [{'label': cls, 'value': cls} for cls in self.data['classifier'].unique()]
        self.generator_options = [{'label': gen, 'value': gen} for gen in self.data['generator'].unique()]
        self.dataset_type_options = [{'label': dt, 'value': dt} for dt in self.data['dataset_type'].unique()]
        
        # Opções para agrupamento
        self.grouping_options = [
            {'label': 'Dataset', 'value': 'dataset'},
            {'label': 'Classificador', 'value': 'classifier'},
            {'label': 'Generator', 'value': 'generator'},
            {'label': 'Tipo de Dataset', 'value': 'dataset_type'}
        ]
        
        # Configura os callbacks
        self.setup_callbacks(app)

    def create_layout(self):
        return html.Div([
            html.H1("Visualização das Métricas de Classificação", style={'textAlign': 'center'}),

            # Dropdowns para seleção
            html.Div([
                html.Label("Selecione o Dataset"),
                dcc.Dropdown(id='dataset-dropdown', options=self.dataset_options, multi=True, value=self.data['dataset'].unique().tolist()),

                html.Label("Selecione os Classificadores"),
                dcc.Dropdown(id='classifier-dropdown', options=self.classifier_options, multi=True, value=self.data['classifier'].unique().tolist()),

                html.Label("Selecione o Generator"),
                dcc.Dropdown(id='generator-dropdown', options=self.generator_options, multi=True, value=self.data['generator'].unique().tolist()),

                html.Label("Selecione o Tipo de Dataset"),
                dcc.Dropdown(id='dataset-type-dropdown', options=self.dataset_type_options, multi=True, value=self.data['dataset_type'].unique().tolist()),

                html.Label("Agrupar por"),
                dcc.Dropdown(id='grouping-dropdown', options=self.grouping_options, value='dataset')  # Seleção de agrupamento
            ], style={'width': '20%', 'padding': '5px', 'float': 'left'}),

            # Gráfico para exibir as comparações de métricas
            html.Div(id='graphs-container', style={'width': '70%', 'float': 'right'})
        ])

    def update_graphs(self, selected_datasets, selected_classifiers, selected_generators, selected_dataset_types, selected_grouping):
        # Filtra os dados com base nas seleções
        filtered_data = self.data[
            (self.data['dataset'].isin(selected_datasets)) &
            (self.data['classifier'].isin(selected_classifiers)) &
            (self.data['generator'].isin(selected_generators)) &
            (self.data['dataset_type'].isin(selected_dataset_types))
        ]
        
        # Verifica se há dados filtrados após a seleção
        if filtered_data.empty:
            return html.Div("Nenhum dado corresponde à seleção. Tente diferentes combinações.")

        # Lista para armazenar os gráficos
        graphs = []

        # Para cada métrica, cria um gráfico separado
        for metric in self.metrics:
            traces = []
            annotations = []

            # Agrupamento pelo critério selecionado
            grouped_data = filtered_data.groupby(selected_grouping)
            
            for group_name, group_df in grouped_data:
                metric_values = group_df[metric].tolist()
                combination_labels = group_df.apply(
                    lambda row: f"{row['dataset']} - {row['classifier']} - {row['generator']} - {row['dataset_type']}", axis=1
                ).tolist()

                # Encontrar o valor máximo para o dataset atual e adicionar uma anotação
                max_value = group_df[metric].max()
                max_index = group_df[metric].idxmax()
                max_label = combination_labels[group_df.index.get_loc(max_index)]
                
                # Define a cor da anotação com base no tipo de dataset
                dataset_type = group_df.loc[max_index, 'dataset_type']
                annotation_color = 'green' if dataset_type in ['mixed', 'synthetic'] else 'blue'
                annotation_color = 'orange' if 'comb' in dataset_type else annotation_color

                # Extrai as iniciais do generator ou usa "TRAIN" se o dataset_type for "train"
                generator_name = group_df.loc[max_index, 'generator']
                generator_initials = "" if dataset_type == "real" else ''.join([word[:3].upper() for word in generator_name.split()])

                # Anotação para o valor máximo do dataset atual
                annotations.append({
                    'x': max_label,
                    'y': max_value,
                    'xref': 'x',
                    'yref': 'y',
                    'text': f"{generator_initials} Máx: {max_value:.2f}",
                    'showarrow': True,
                    'arrowhead': 7,
                    'ax': 0,
                    'ay': -40,
                    'font': {'color': annotation_color, 'size': 12}
                })

                # Cria o trace para a métrica
                traces.append(go.Bar(
                    x=combination_labels,
                    y=metric_values,
                    name=f"{group_name}"
                ))

            # Ajuste o layout da figura dependendo do número de combinações no eixo X
            fig_width = max(1300, len(filtered_data[selected_grouping].unique()) * 60)

            # Layout do gráfico com anotação para cada dataset
            figure = {
                'data': traces,
                'layout': go.Layout(
                    title=f"Comparação de {metric.capitalize()}",
                    xaxis={'title': f'Combinações (Agrupado por {selected_grouping.capitalize()})', 'automargin': True, 'tickangle': -45},
                    yaxis={'title': 'Valor'},
                    barmode='group',
                    showlegend=True,
                    width=fig_width,
                    height=800,
                    annotations=annotations  # Adiciona as anotações de valor máximo
                )
            }

            # Adiciona o gráfico ao container
            graphs.append(dcc.Graph(figure=figure))

        # Se não houver gráficos para exibir
        if not graphs:
            return html.Div("Não há gráficos disponíveis para os dados selecionados.")

        return graphs

    def setup_callbacks(self, app):
        @app.callback(
            Output('graphs-container', 'children'),
            [
                Input('dataset-dropdown', 'value'),
                Input('classifier-dropdown', 'value'),
                Input('generator-dropdown', 'value'),
                Input('dataset-type-dropdown', 'value'),
                Input('grouping-dropdown', 'value')
            ]
        )
        def update_graphs(selected_datasets, selected_classifiers, selected_generators, selected_dataset_types, selected_grouping):
            return self.update_graphs(selected_datasets, selected_classifiers, selected_generators, selected_dataset_types, selected_grouping)
