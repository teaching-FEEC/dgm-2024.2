import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def visualize_tsne_r_s(
    X: np.ndarray,
    y: np.ndarray,
    X_gen: np.ndarray,
    y_gen: np.ndarray,
    path: str = "/tmp/tsne_embeddings.pdf",
    feature_averaging: bool = False,
    perplexity: float = 30.0
) -> None:
    """
    Visualizes t-SNE embeddings of real and synthetic data.

    This function generates a scatter plot of t-SNE embeddings for real and synthetic data.
    Each data point is represented by a marker on the plot, and the colors of the markers
    correspond to the corresponding class labels of the data points.

    :param X: The original real data tensor of shape (num_samples, num_features).
    :type X: np.ndarray
    :param y: The labels of the original real data tensor of shape (num_samples,).
    :type y: np.ndarray
    :param X_gen: The generated synthetic data tensor of shape (num_samples, num_features).
    :type X_gen: np.ndarray
    :param y_gen: The labels of the generated synthetic data tensor of shape (num_samples,).
    :type y_gen: np.ndarray
    :param path: The path to save the visualization as a PDF file. Defaults to "/tmp/tsne_embeddings.pdf".
    :type path: str, optional
    :param feature_averaging: Whether to compute the average features for each class. Defaults to False.
    :type feature_averaging: bool, optional
    :param perplexity: Perplexity parameter for t-SNE.
    :type perplexity: float, optional
    """
    # Inicializar t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='random')
    
    # Preparar dados para t-SNE
    if feature_averaging:
        X_all = np.concatenate((np.mean(X, axis=2), np.mean(X_gen, axis=2)))
    else:
        X_all = np.concatenate((X, X_gen), axis=0)
    
    X_emb = tsne.fit_transform(X_all.reshape(X_all.shape[0], -1))
    
    # Preparar labels
    y_all = np.concatenate((y, y_gen), axis=0)
    labels = np.concatenate((np.zeros(len(y)), np.ones(len(y_gen))), axis=0)  # 0: real, 1: generated
    
    # Criar DataFrame para seaborn
    df = pd.DataFrame({
        'TSNE1': X_emb[:, 0],
        'TSNE2': X_emb[:, 1],
        'Type': ['Real'] * len(y) + ['Generated'] * len(y_gen),
        'Label': labels
    })

    # Definir paleta de cores e estilos
    palette = {'Real': 'blue', 'Generated': 'red'}
    markers = {"Real": "<", "Generated": "H"}

    # Plotar usando seaborn
    plt.figure(figsize=(8, 6), dpi=80)
    ax = sns.scatterplot(
        data=df,
        x='TSNE1',
        y='TSNE2',
        hue='Type',
        style='Type',
        markers=markers,
        palette=palette,
        alpha=0.7
    )
    
    # Ajustar legenda
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, title="Type", fontsize=12, title_fontsize=14, loc='best', bbox_to_anchor=(1.05, 1))
    
    # Ajustar a largura das linhas na legenda
    for handle in handles:
        if isinstance(handle, plt.Line2D):
            handle.set_linewidth(2.0)
        elif isinstance(handle, plt.PathCollection):
            handle.set_edgecolor('black')
            handle.set_linewidth(2.0)

    plt.title('t-SNE Visualization of Real and Generated Data')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.box(False)
    plt.axis("off")
    plt.savefig(path, bbox_inches='tight')
    plt.show()

# Exemplo de chamada da função
#visualize_tsne(X_train, y, X_gen, y_gen)
