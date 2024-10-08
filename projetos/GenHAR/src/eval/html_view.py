import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import spectrogram, correlate
import base64
import io

# Dicionário de datasets
datasets = {}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Análise de Dados de Sensores"),

    # Upload button to add datasets
    dcc.Upload(
        id='upload-data',
        children=html.Button('Carregar Dataset (CSV)'),
        multiple=False  # Permitir apenas um arquivo por vez
    ),
    
    # Dropdowns for dataset selection
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[],  # Inicialmente vazio
        value=None,
        multi=True,
        placeholder='Selecione um ou mais datasets'
    ),

    dcc.Dropdown(
        id='sample-dropdown',
        multi=True,  # Permitir múltiplas seleções de amostras
        placeholder='Selecione uma ou mais amostras'
    ),

    dcc.Dropdown(
        id='axis-dropdown',
        placeholder='Selecione um eixo',
        multi=True, 
    ),

    dcc.Graph(id='combined-plot')
])

@app.callback(
    Output('sample-dropdown', 'options'),
    Output('sample-dropdown', 'value'),
    Output('axis-dropdown', 'options'),
    Output('axis-dropdown', 'value'),
    Input('dataset-dropdown', 'value')
)
def update_sample_and_axis_options(selected_datasets):
    if selected_datasets and selected_datasets[0] in datasets:
        df = datasets[selected_datasets[0]]        
        n_samples = df.shape[0]
        n_sensors = df.shape[1]

        sample_options = [{'label': f'Sample {i}', 'value': i} for i in range(n_samples)]
        axis_options = [{'label': f'Sensor {i}', 'value': i} for i in range(n_sensors)]

        return sample_options, [0], axis_options, [0]
    return [], None, [], None

@app.callback(
    Output('combined-plot', 'figure'),
    Input('sample-dropdown', 'value'),
    Input('axis-dropdown', 'value'),
    Input('dataset-dropdown', 'value')
)
def update_graphs(selected_samples, selected_axis, selected_datasets):
    if not selected_datasets or not selected_samples:
        return go.Figure()

    num_datasets = len(selected_datasets)
    num_samples = len(selected_samples)

    # Criar subplots: 4 linhas para 4 gráficos, colunas igual ao número de datasets
    fig = make_subplots(
        rows=4, 
        cols=num_datasets * num_samples, 
        subplot_titles=[f'{dataset_name} - Sample {sample_idx}' for dataset_name in selected_datasets for sample_idx in selected_samples]
    )

    for idx, dataset_name in enumerate(selected_datasets):
        df_ = datasets[dataset_name]
        df = df_.drop(columns='label')
        n_amostras = df.shape[0]
        data_r = df.values.reshape(n_amostras, 6, 60)
        
        for sample_idx in selected_samples:
            sample_data = data_r[sample_idx]

            # Calcular a posição correta da coluna para o dataset e amostra
            col_position = idx * num_samples + selected_samples.index(sample_idx) + 1
            for sensor in selected_axis:
                sample_data = data_r[sample_idx][sensor]
                # (a) Gráfico da série temporal
                fig.add_trace(go.Scatter(
                    x=np.arange(len(sample_data)),
                    y=sample_data,
                    mode='lines+markers',
                    name=f'{dataset_name} - {sensor} Sample {sample_idx}'
                ), row=1, col=col_position)

            for sensor in selected_axis:
                sample_data = data_r[sample_idx][sensor]
                # (b) Espectro de magnitude
                freq = np.fft.rfftfreq(len(sample_data), d=1)
                spectrum = np.abs(np.fft.rfft(sample_data))

                fig.add_trace(go.Scatter(
                    x=freq,
                    y=spectrum,
                    mode='lines',
                    name=f'{dataset_name} - Sample {sample_idx}'
                ), row=2, col=col_position)

            # (c) Função de Autocorrelação
            for sensor in selected_axis:
                sample_data = data_r[sample_idx][sensor]

                lags = np.arange(len(sample_data))
                autocorr = correlate(sample_data, sample_data, mode='full')

                fig.add_trace(go.Scatter(
                    x=lags - len(sample_data) + 1,
                    y=autocorr,
                    mode='lines',
                    name=f'{dataset_name} - Sample {sample_idx}'
                ), row=3, col=col_position)

            y_offsets = []

            # Calcula os deslocamentos para que os heatmaps sejam concatenados verticalmente
            max_f = 0  # Valor inicial para calcular o deslocamento

            for sensor in selected_axis:
                # Obtém os dados da amostra para o sensor atual
                sample_data = data_r[sample_idx][sensor]
                
                # Gera o espectrograma da amostra
                f, t, Sxx = spectrogram(sample_data, nperseg=30)
                
                # Ajusta o eixo y (frequências) para concatenar os heatmaps verticalmente
                y_adjusted = f + max_f
                max_f += max(f)  # Atualiza o deslocamento máximo de y para o próximo sensor

                # Adiciona o espectrograma como um heatmap, ajustando y
                fig.add_trace(go.Heatmap(
                    z=Sxx,              # Dados do espectrograma
                    x=t,                # Tempo
                    y=y_adjusted,       # Frequências ajustadas para evitar sobreposição
                    colorscale='Viridis',  # Escala de cores do heatmap
                    showscale=False,    # Oculta a barra de escala
                    name=f'{dataset_name} - Sample {sample_idx} - Sensor {sensor}'
                ), row=4, col=col_position)

    # Atualizar o layout da figura
    fig.update_layout(title='Análise de Dados de Sensores', height=800)
    fig.update_xaxes(title_text="Tempo", row=1, col=1)
    fig.update_yaxes(title_text="Valor", row=1, col=1)
    fig.update_xaxes(title_text="Frequência", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_xaxes(title_text="Lags", row=3, col=1)
    fig.update_yaxes(title_text="Autocorrelação", row=3, col=1)
    fig.update_xaxes(title_text="Tempo", row=4, col=1)
    fig.update_yaxes(title_text="Frequência", row=4, col=1)

    return fig

@app.callback(
    Output('dataset-dropdown', 'options'),
    Input('upload-data', 'contents')
)
def upload_dataset(contents):
    if contents is not None:
        # Decode the contents and read it into a DataFrame
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df_x = df.drop(columns='label')
        
        # Add the new dataset to the dictionary
        dataset_name = f'Dataset {len(datasets) + 1}'  # Create a new name
        datasets[dataset_name] = df
        
        # Update the dropdown options
        return [{'label': name, 'value': name} for name in datasets.keys()]
    
    # Return existing options if no new dataset is uploaded
    return [{'label': name, 'value': name} for name in datasets.keys()]

if __name__ == '__main__':
    app.run_server(debug=True)
