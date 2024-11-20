from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
from dash import Dash

class PrivacyBasedMetrics:
    def __init__(self, file_dbm_metrics, app):
        self.data = pd.read_csv(file_dbm_metrics)
        self.metrics = ['EXAC_MATCH', 'NEIG_PRIV_SCORE', 'MEM_INF_SCORE']        
        self.dataset_options = [{'label': ds, 'value': ds} for ds in self.data['dataset'].unique()]
        self.generator_options = [{'label': gen, 'value': gen} for gen in self.data['generator'].unique()]       
        self.grouping_options = [
            {'label': 'Dataset', 'value': 'dataset'},
            {'label': 'Generator', 'value': 'generator'}
        ]
        self.setup_callbacks(app)

    def create_layout(self):
        return html.Div([
            html.H1("Visualização de Métricas de Privacidade", style={'textAlign': 'center'}),

            # Dropdowns de seleção
            html.Div([
                html.Label("Selecione o Dataset"),
                dcc.Dropdown(id='dataset-dropdown_privacy', options=self.dataset_options, multi=True, value=self.data['dataset'].unique().tolist()),

                html.Label("Selecione o Generator"),
                dcc.Dropdown(id='generator-dropdown_privacy', options=self.generator_options, multi=True, value=self.data['generator'].unique().tolist()),

                html.Label("Agrupar por"),
                dcc.Dropdown(id='grouping-dropdown_privacy', options=self.grouping_options, value='dataset')
            ], style={'width': '20%', 'padding': '5px', 'float': 'left'}),

            # Explicações das métricas
            html.Div([
    html.H3("Explicações das Métricas", style={'textAlign': 'center'}),
    html.P("https://aws.amazon.com/blogs/machine-learning/how-to-evaluate-the-quality-of-the-synthetic-data-measuring-from-the-perspective-of-fidelity-utility-and-privacy/"),
    html.P("As métricas apresentadas são as seguintes:"),
    html.Ul([
        html.Li("EXAC_MATCH: Mede a exatidão da correspondência entre amostras. Valores mais altos indicam maior precisão, mas podem comprometer a privacidade, já que tornam mais fácil identificar dados sensíveis."),
        html.Li("NEIG_PRIV_SCORE: Mede a privacidade em relação a vizinhos próximos. Valores mais baixos indicam maior privacidade, pois dificultam a associação entre amostras similares."),
        html.Li("MEM_INF_SCORE: Mede a proteção de privacidade com base em informações memorizadas. Valores mais baixos indicam maior proteção de privacidade, limitando a recuperação de dados sensíveis.")
    ]),
    html.H3("Intervalos de Aceitação", style={'textAlign': 'center'}),
    html.P("Os intervalos de aceitação para cada métrica são os seguintes:"),
    html.Ul([
        html.Li("EXAC_MATCH: Idealmente acima de 0.95 para garantir uma boa correspondência, mas sem comprometer a privacidade."),
        html.Li("NEIG_PRIV_SCORE: Abaixo de 0.5 é considerado bom, pois significa que o modelo preserva bem a privacidade, dificultando a identificação de amostras semelhantes."),
        html.Li("MEM_INF_SCORE: Abaixo de 0.5 é considerado bom, pois indica que o modelo não está memorizando informações sensíveis que poderiam comprometer a privacidade.")
    ]),
]

, style={'width': '70%', 'float': 'right', 'padding': '20px'}),
            
            # Container dos gráficos
            html.Div(id='graphs-container_privacy', style={'width': '70%', 'float': 'right'})
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
            'EXAC_MATCH': 0.95,
            'NEIG_PRIV_SCORE': 0.5,
            'MEM_INF_SCORE': 0.5,
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
                    title=f"Comparação de {metric} (Privacidade)",
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
            Output('graphs-container_privacy', 'children'),
            [
                Input('dataset-dropdown_privacy', 'value'),
                Input('generator-dropdown_privacy', 'value'),
                Input('grouping-dropdown_privacy', 'value')
            ]
        )
        def update_graphs(selected_datasets, selected_generators, selected_grouping):
            return self.update_graphs(selected_datasets, selected_generators, selected_grouping)
