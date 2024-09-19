import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objs as go

from standartized_balanced import StandardizedBalancedDataset



# Função para carregar os dados
def get_data(dataset_name, sensors, normalize_data):
    data_folder = f"/HDD/dados/amparo/meta4/M4-Framework-Experiments/experiments/experiment_executor/data/standartized_balanced/{dataset_name}/"
    dataset = StandardizedBalancedDataset(data_folder, sensors=sensors)
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_all_data(normalize_data=normalize_data, resize_data=False)
    return X_train, y_train, X_test, y_test, X_val, y_val




dataset="UCI"
transform="time"
# Carregar os dados originais
X_train, y_train, X_test, y_test, X_val, y_val = get_data(dataset, ['accel', 'gyro'], False)

# Função para aplicar FFT
def apply_fft(X):
    print("fft")
    return np.abs(np.fft.fft(X, axis=-1))
    

if transform=='fft':
    X_train = apply_fft(X_train)
    #X_test = apply_fft(X_test)

# Definir o diretório onde os arquivos CSV dos dados gerados estão armazenados
csv_directory = f"{dataset}/{transform}"

# Listar todos os arquivos CSV no diretório
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Lista para armazenar os dados carregados
all_data = []
labels = []

# Carregar todos os arquivos CSV e armazenar os dados em uma lista
for file in csv_files:
    file_path = os.path.join(csv_directory, file)
    df = pd.read_csv(file_path)
    class_label = int(file.split('_')[-1].split('.')[0])  # Extrair a classe do nome do arquivo
    all_data.append(df.iloc[:, :-1])  # Assume que a última coluna seja a classe
    labels.append([class_label + 6] * len(df))  # Adicionar 6 para distinguir dos rótulos das classes originais

# Concatenar todos os dados gerados e labels
all_data = pd.concat(all_data, ignore_index=True)
labels = [item for sublist in labels for item in sublist]

# Preparar os dados originais para t-SNE
X_train_sample = X_train.reshape(X_train.shape[0], -1)
y_train_sample = y_train
print("X train",X_train_sample.shape)
print("sintetic",all_data.shape)

# Combinar dados originais e gerados
combined_data = np.concatenate((X_train_sample, all_data), axis=0)
combined_labels = np.concatenate((y_train_sample, labels), axis=0)


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Train SVM classifier
clf = SVC()
clf.fit(combined_data, combined_labels)
        
        # Predict and evaluate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print(f'{dataset} Accuracy with : {accuracy:.2f}')





# Aplicar t-SNE para reduzir as dimensões dos dados combinados
tsne = TSNE(n_components=2, random_state=42)
reduced_combined_data = tsne.fit_transform(combined_data)

# Criar um DataFrame com os dados reduzidos e os labels
df_combined = pd.DataFrame(reduced_combined_data, columns=['Component 1', 'Component 2'])
df_combined['Class'] = combined_labels

# Criar traços para cada uma das 12 classes
scatter_data = []

for class_label in np.unique(combined_labels):
    trace = go.Scatter(
        x=df_combined[df_combined['Class'] == class_label]['Component 1'],
        y=df_combined[df_combined['Class'] == class_label]['Component 2'],
        mode='markers',
        name=f'Class {class_label}' if class_label < 6 else f'Generated Class {class_label-6}',
        marker=dict(size=5),
        showlegend=True
    )
    scatter_data.append(trace)

# Configurar o layout do gráfico
layout = go.Layout(
    title=f'{dataset} {transform}_t-SNE Scatter Plot of Original and Generated Data',
    xaxis=dict(title='Component 1'),
    yaxis=dict(title='Component 2')
)

# Criar a figura e adicionar os traços
fig = go.Figure(data=scatter_data, layout=layout)

# Salvar o gráfico em um arquivo HTML
fig.write_html(f"{dataset}/{transform}tsne_scatter_plot_original_and_generated.html")

# Mostrar o gráfico
fig.show()
