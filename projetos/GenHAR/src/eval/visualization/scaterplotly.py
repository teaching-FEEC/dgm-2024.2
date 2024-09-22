import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import plotly.colors as pc

def reduce_data(x_real,x_gen,y,y_gen):
    from sklearn.manifold import TSNE
    import pandas as pd
    import numpy as np
    import plotly.graph_objs as go
    #y_labels = np.argmax(y, axis=1)
    y_labels=y
    y_gen_labels = y_gen #np.argmax(y_gen, axis=1)
    transform="None"
    # Combinar dados originais e gerados
    combined_data = np.concatenate((x_real, x_gen), axis=0)
    combined_labels = np.concatenate((y_labels, y_gen_labels), axis=0)
    print(combined_data.shape)

    #combined_data = combined_data.reshape(combined_data.shape[0], combined_data.shape[1] * combined_data.shape[2])

    tsne = TSNE(n_components=2, random_state=42)
    reduced_combined_data = tsne.fit_transform(combined_data)

    # Criar um DataFrame com os dados reduzidos e os labels
    df_combined = pd.DataFrame(reduced_combined_data, columns=['Component 1', 'Component 2'])
    df_combined['Class'] = combined_labels
    return df_combined,combined_labels


def get_plotly_by_labels(x_real,x_gen,y_labels,y_gen_labels,dataset="",transform=""):

    df_combined,combined_labels=reduce_data(x_real,x_gen,y_labels,y_gen_labels)
    
    # Definir os símbolos para as classes reais e geradas
    real_symbol = 'circle'       # Símbolo para classes reais
    generated_symbol = 'diamond'  # Símbolo para classes geradas

    # Obter o número único de classes
    unique_labels = np.unique(combined_labels)
    num_classes = len(unique_labels)

    colors = pc.qualitative.Dark24 if num_classes <= 24 else pc.sample_colorscale('Viridis', num_classes)

    # Criar um dicionário que associa cada classe a uma cor
    class_colors = {class_label: colors[i % len(colors)] for i, class_label in enumerate(unique_labels)}

    # Criar traços para cada classe
    scatter_data = []

    # Iterar sobre cada classe única nas labels
    for class_label in unique_labels:
        color = class_colors[class_label]  # Atribuir a cor específica da classe
        
        # Dados Reais (X_train)
        trace_real = go.Scatter(
            x=df_combined[(combined_labels == class_label) & (np.arange(len(combined_labels)) < len(y_labels))]['Component 1'],
            y=df_combined[(combined_labels == class_label) & (np.arange(len(combined_labels)) < len(y_labels))]['Component 2'],
            mode='markers',
            name=f'Real Class {class_label}',
            marker=dict(size=5, symbol=real_symbol, color=color),  # Mesma cor, símbolo diferente
            showlegend=True
        )
        scatter_data.append(trace_real)

        # Dados Gerados (X_gen)
        trace_gen = go.Scatter(
            x=df_combined[(combined_labels == class_label) & (np.arange(len(combined_labels)) >= len(y_labels))]['Component 1'],
            y=df_combined[(combined_labels == class_label) & (np.arange(len(combined_labels)) >= len(y_labels))]['Component 2'],
            mode='markers',
            name=f'Generated Class {class_label}',
            marker=dict(size=5, symbol=generated_symbol, color=color),  # Mesma cor, símbolo diferente
            showlegend=True
        )
        scatter_data.append(trace_gen)

    # Configurar o layout do gráfico
    layout = go.Layout(
        title=f'{dataset} {transform}_t-SNE Scatter Plot of Original and Generated Data',
        xaxis=dict(title='Component 1'),
        yaxis=dict(title='Component 2')
    )

    # Criar a figura e adicionar os traços
    fig = go.Figure(data=scatter_data, layout=layout)

    # Mostrar o gráfico
    fig.show()

    

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
from sklearn.manifold import TSNE

def tsne_subplots_by_labels(x_real, y_real, x_gen, y_gen, label_names=None):
    # Reshape if 3D
    if x_real.ndim == 3:
        x_real = x_real.reshape(x_real.shape[0], -1)
    if x_gen.ndim == 3:
        x_gen = x_gen.reshape(x_gen.shape[0], -1)

    if y_real.ndim > 1:
        y_real = np.argmax(y_real, axis=1)
    if y_gen.ndim > 1:
        y_gen = np.argmax(y_gen, axis=1)

    # Combine data for t-SNE
    combined_data = np.concatenate((x_real, x_gen))
    combined_labels = np.concatenate((y_real, y_gen))

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(combined_data)

    # Create DataFrame for plotting
    df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])
    df['Label'] = combined_labels
    df['Source'] = ['Real'] * len(y_real) + ['Generated'] * len(y_gen)

    unique_labels = np.unique(combined_labels)
    num_labels = len(unique_labels)

    # Create subplots
    cols = 2
    rows = (num_labels + 1) // cols
    fig = sp.make_subplots(rows=rows, cols=cols, 
                           subplot_titles=[label_names[label] if label_names else f'Label {label}' for label in unique_labels])

    for i, label in enumerate(unique_labels):
        row = i // cols + 1
        col = i % cols + 1
        
        # Filter data for the specific label and source
        label_data_real = df[(df['Label'] == label) & (df['Source'] == 'Real')]
        label_data_gen = df[(df['Label'] == label) & (df['Source'] == 'Generated')]

        # Add traces for real data
        fig.add_trace(
            px.scatter(label_data_real, x='Component 1', y='Component 2', color_discrete_sequence=['blue']).data[0],
            row=row, col=col
        )

        # Add traces for generated data
        fig.add_trace(
            px.scatter(label_data_gen, x='Component 1', y='Component 2', color_discrete_sequence=['red']).data[0],
            row=row, col=col
        )

    fig.update_layout(title='t-SNE Visualization of Real and Generated Data by Label', showlegend=False)
    fig.show()
