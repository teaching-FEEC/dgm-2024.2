from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import umap


def plot_umap_by_label(dataset, transf, X, labels, class_names):
    # Inicializa o modelo UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean")
    umap_embeddings = umap_model.fit_transform(X)

    # Cria a figura e os eixos
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plota o gráfico de dispersão
    scatter = ax.scatter(
        umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7
    )

    # Adiciona a legenda
    handles, _ = scatter.legend_elements()
    ax.legend(handles=handles, labels=class_names, title="Classes")

    # Configurações dos eixos
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{dataset} {transf} UMAP Projection")

    # Retorna a figura
    return fig


def split_axis_reshape(data):
    n_amostras = data.shape[0]
    return data.values.reshape(n_amostras, 6, 60)


def dict_class_samples(data, labels):
    class_data = defaultdict(list)
    for X, y in zip(data, labels):
        class_data[y].append(X)
    for label in class_data.keys():
        class_data[label] = np.array(class_data[label])
    return class_data
