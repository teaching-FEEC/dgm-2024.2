from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
from dash import Dash

class DistanceBasedMeasures:
    def __init__(self, file_dbm_metrics, app):
        # Carregar os dados e configurar métricas
        self.data = pd.read_csv(file_dbm_metrics)
        self.metrics = ['ED', 'DTW', 'Minkowski', 'Manhattan', 'Cosine', 'Pearson']
        
        # Opções para dropdowns
        self.dataset_options = [{'label': ds, 'value': ds} for ds in self.data['dataset'].unique()]
        self.generator_options = [{'label': gen, 'value': gen} for gen in self.data['generator'].unique()]
        
        # Opções para agrupamento
        self.grouping_options = [
            {'label': 'Dataset', 'value': 'dataset'},
            {'label': 'Generator', 'value': 'generator'}
        ]
        
        # Configurar callbacks
        self.setup_callbacks(app)

    def create_layout(self):
        return html.Div([
            html.H1("Visualização de Medidas Baseadas em Distância", style={'textAlign': 'center'}),

            # Dropdowns de seleção
            html.Div([
                html.Label("Selecione o Dataset"),
                dcc.Dropdown(id='dataset-dropdown_dbm', options=self.dataset_options, multi=True, value=self.data['dataset'].unique().tolist()),

                html.Label("Selecione o Generator"),
                dcc.Dropdown(id='generator-dropdown_dbm', options=self.generator_options, multi=True, value=self.data['generator'].unique().tolist()),

                html.Label("Agrupar por"),
                dcc.Dropdown(id='grouping-dropdown_dbm', options=self.grouping_options, value='dataset')
            ], style={'width': '20%', 'padding': '5px', 'float': 'left'}),

            # Explicações das métricas
            html.Div([
                html.H3("Explicações das Métricas", style={'textAlign': 'center'}),
                html.P("As métricas apresentadas são as seguintes:"),
                html.Ul([
                    html.Li("ED (Euclidean Distance): Mede a distância euclidiana entre duas amostras. Idealmente, valores baixos indicam maior similaridade."),
                    html.Li("DTW (Dynamic Time Warping): Mede a similaridade entre duas séries temporais, ajustando-as temporalmente. Valores mais baixos indicam maior similaridade."),
                    html.Li("Minkowski: Generaliza as distâncias Euclidiana e Manhattan, com valores mais baixos indicando maior similaridade."),
                    html.Li("Manhattan: Mede a soma das distâncias absolutas entre pontos. Valores baixos indicam maior similaridade."),
                    html.Li("Cosine: Mede a similaridade de orientação entre vetores. Valores próximos de 1 indicam maior similaridade."),
                    html.Li("Pearson: Mede a correlação linear entre variáveis. Valores próximos de 1 indicam correlação positiva.")
                ]),
                html.H3("Intervalos de Aceitação", style={'textAlign': 'center'}),
                html.P("Os intervalos de aceitação para cada métrica são os seguintes:"),
                html.Ul([
                    html.Li("ED: Idealmente abaixo de 0.5 para boa similaridade."),
                    html.Li("DTW: Abaixo de 0.7 é considerado bom."),
                    html.Li("Minkowski: Valor ideal depende do parâmetro p; em geral, abaixo de 1 é considerado bom."),
                    html.Li("Manhattan: Idealmente abaixo de 1 para boa similaridade."),
                    html.Li("Cosine: Próximo de 1 indica alta similaridade."),
                    html.Li("Pearson: Próximo de 1 indica alta correlação.")
                ]),
            ], style={'width': '70%', 'float': 'right', 'padding': '20px'}),
            
            # Container dos gráficos
            html.Div(id='graphs-container_dbm', style={'width': '70%', 'float': 'right'})
        ])

    def update_graphs(self, selected_datasets, selected_generators, selected_grouping):
        # Filtrar os dados com base nas seleções
        filtered_data = self.data[
            (self.data['dataset'].isin(selected_datasets)) &
            (self.data['generator'].isin(selected_generators))
        ]
        
        # Verificar se há dados filtrados após a seleção
        if filtered_data.empty:
            return html.Div("Nenhum dado corresponde à seleção. Tente diferentes combinações.")

        # Lista para armazenar gráficos
        graphs = []

        # Definir os limites de aceitação para cada métrica
        metric_limits = {
            'ED': 0.5,
            'DTW': 0.7,
            'Minkowski': 1.0,
            'Manhattan': 1.0,
            'Cosine': 0.95,
            'Pearson': 0.9,
        }

        # Para cada métrica, cria um gráfico separado
        for metric in self.metrics:
            traces = []
            annotations = []
            shapes = []

            # Agrupar pelo critério selecionado
            grouped_data = filtered_data.groupby(selected_grouping)
            
            for group_name, group_df in grouped_data:
                metric_values = group_df[metric].tolist()
                combination_labels = group_df.apply(
                    lambda row: f"DBM - {row['dataset']} - {row['generator']}", axis=1
                ).tolist()

                # Encontra o valor máximo para o dataset atual e adiciona uma anotação
                min_value = group_df[metric].min()
                min_index = group_df[metric].idxmin()
                min_label = combination_labels[group_df.index.get_loc(min_index)]
                
                # Define a cor da anotação com base no generator
                generator_name = group_df.loc[min_index, 'generator']
                annotation_color = 'purple' if "synth" in generator_name.lower() else 'orange'

                # Anotação para o valor máximo do dataset atual
                annotations.append({
                    'x': min_label,
                    'y': min_value,
                    'xref': 'x',
                    'yref': 'y',
                    'text': f"Min: {min_value:.2f}",
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

            # Obter o limite específico para a métrica
            limit_value = metric_limits.get(metric, 0.5)

            # Adiciona uma linha horizontal no gráfico com o limite de aceitação
            shapes.append({
                'type': 'line',
                'x0': -0.5,
                'x1': len(filtered_data[selected_grouping].unique()),
                'y0': limit_value,
                'y1': limit_value,
                'line': {
                    'color': 'red',
                    'width': 2,
                    'dash': 'dash'
                }
            })

            # Ajuste o layout da figura dependendo do número de combinações no eixo X
            fig_width = max(1300, len(filtered_data[selected_grouping].unique()) * 60)

            # Layout do gráfico com anotação para cada dataset
            figure = {
                'data': traces,
                'layout': go.Layout(
                    title=f"Comparação de {metric} (DBM)",
                    xaxis={'title': f'Combinações (Agrupado por {selected_grouping.capitalize()})', 'automargin': True, 'tickangle': -45},
                    yaxis={'title': 'Valor', 'range': [0, max(filtered_data[metric].max(), limit_value * 1.5)]},
                    barmode='group',
                    showlegend=True,
                    width=fig_width,
                    height=800,
                    annotations=annotations,
                    shapes=shapes
                )
            }

            # Adiciona o gráfico ao container
            graphs.append(dcc.Graph(figure=figure))

        if not graphs:
            return html.Div("Não há gráficos disponíveis para os dados selecionados.")

        return graphs

    def setup_callbacks(self, app):
        @app.callback(
            Output('graphs-container_dbm', 'children'),
            [
                Input('dataset-dropdown_dbm', 'value'),
                Input('generator-dropdown_dbm', 'value'),
                Input('grouping-dropdown_dbm', 'value')
            ]
        )
        def update_graphs(selected_datasets, selected_generators, selected_grouping):
            return self.update_graphs(selected_datasets, selected_generators, selected_grouping)
