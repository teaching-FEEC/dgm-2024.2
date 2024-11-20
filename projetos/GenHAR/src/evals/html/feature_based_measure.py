from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
from dash import Dash

class FeatureBasedMeasures:
    def __init__(self, file_fbm_metrics, app):
        # Carregar os dados e configurar métricas
        self.data = pd.read_csv(file_fbm_metrics)
        self.metrics = ['MDD', 'ACD', 'SD', 'KD']
        
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
            html.H1("Visualização de Medidas Baseadas em Características", style={'textAlign': 'center'}),

            # Dropdowns de seleção
            html.Div([
                html.Label("Selecione o Dataset"),
                dcc.Dropdown(id='dataset-dropdown_fbm', options=self.dataset_options, multi=True, value=self.data['dataset'].unique().tolist()),

                html.Label("Selecione o Generator"),
                dcc.Dropdown(id='generator-dropdown_fbm', options=self.generator_options, multi=True, value=self.data['generator'].unique().tolist()),

                html.Label("Agrupar por"),
                dcc.Dropdown(id='grouping-dropdown_fbm', options=self.grouping_options, value='dataset')  # Agrupamento padrão por 'dataset'
            ], style={'width': '20%', 'padding': '5px', 'float': 'left'}),

            # Explicações das métricas
            html.Div([
                html.H3("Explicações das Métricas", style={'textAlign': 'center'}),
                html.P("As métricas apresentadas são as seguintes:"),
                html.Ul([
                    html.Li("MDD (Mean Distance Deviation): Mede a distância média de desvio entre os dados originais e os dados transformados. Este valor idealmente deve ser baixo, indicando que a transformação preservou bem as características."),
                    html.Li("ACD (Average Consistency Deviation): Mede a consistência média entre as amostras. Quanto menor o valor, mais consistentes são as transformações."),
                    html.Li("SD (Standard Deviation): Mede a dispersão dos valores. Valores mais baixos indicam que os dados estão mais concentrados em torno da média."),
                    html.Li("KD (Kullback-Leibler Divergence): Mede a diferença entre duas distribuições de probabilidade. Valores menores indicam que as distribuições dos dados transformados são mais próximas das originais.")
                ]),
                html.H3("Intervalos de Aceitação", style={'textAlign': 'center'}),
                html.P("Os intervalos de aceitação para cada métrica são os seguintes:"),
                html.Ul([
                    html.Li("MDD: O valor ideal é abaixo de 0.1 para garantir que a transformação preserva bem as características dos dados."),
                    html.Li("ACD: Valores abaixo de 0.05 são aceitáveis para garantir boa consistência."),
                    html.Li("SD: O valor ideal é abaixo de 0.2, indicando baixa dispersão."),
                    html.Li("KD: Valores abaixo de 0.1 indicam uma boa aproximação entre as distribuições.")
                ]),
            ], style={'width': '70%', 'float': 'right', 'padding': '20px'}),
            
            # Container dos gráficos
            html.Div(id='graphs-container_fbm', style={'width': '70%', 'float': 'right'})
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
            'MDD': 0.1,  # Limite para MDD
            'ACD': 0.05,  # Limite para ACD
            'SD': 0.2,  # Limite para SD
            'KD': 0.1,  # Limite para KD
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
                    lambda row: f"FBM - {row['dataset']} - {row['generator']}", axis=1
                ).tolist()

                # Encontrar o valor máximo para o dataset atual e adicionar uma anotação
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
            limit_value = metric_limits.get(metric, 0.2)  # Padrão para 0.2, mas pode ser alterado conforme a métrica

            # Adiciona uma linha horizontal no gráfico com o limite de aceitação
            shapes.append({
                'type': 'line',
                'x0': -0.5,  # Ajuste do valor de x0 para garantir que a linha abranja todo o gráfico
                'x1': len(filtered_data[selected_grouping].unique()),  # Ajuste do valor de x1
                'y0': limit_value,  # Valor da linha no eixo Y
                'y1': limit_value,  # Valor da linha no eixo Y
                'line': {
                    'color': 'red',  # Cor da linha (vermelha)
                    'width': 2,  # Largura da linha
                    'dash': 'dash'  # Tipo da linha (tracejada)
                }
            })

            # Ajuste o layout da figura dependendo do número de combinações no eixo X
            fig_width = max(1300, len(filtered_data[selected_grouping].unique()) * 60)

            # Layout do gráfico com anotação para cada dataset
            figure = {
                'data': traces,
                'layout': go.Layout(
                    title=f"Comparação de {metric} (FBM)",
                    xaxis={'title': f'Combinações (Agrupado por {selected_grouping.capitalize()})', 'automargin': True, 'tickangle': -45},
                    yaxis={'title': 'Valor', 'range': [0, max(filtered_data[metric].max(), limit_value * 1.5)]},  # Ajuste do intervalo do eixo Y
                    barmode='group',
                    showlegend=True,
                    width=fig_width,
                    height=800,
                    annotations=annotations,  # Adiciona as anotações de valor máximo
                    shapes=shapes  # Adiciona a linha de limite
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
            Output('graphs-container_fbm', 'children'),
            [
                Input('dataset-dropdown_fbm', 'value'),
                Input('generator-dropdown_fbm', 'value'),
                Input('grouping-dropdown_fbm', 'value')
            ]
        )
        def update_graphs(selected_datasets, selected_generators, selected_grouping):
            return self.update_graphs(selected_datasets, selected_generators, selected_grouping)
