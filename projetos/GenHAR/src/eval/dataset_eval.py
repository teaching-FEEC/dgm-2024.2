import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
import plotly.express as px

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class TimeSeriesDatasetEvaluator:
    def __init__(self, df, label_col, label_names):
        """
        Classe para avaliar o dataset de séries temporais.

        df: DataFrame que contém os dados de séries temporais.
        label_col: Nome da coluna que contém os rótulos.
        """
        self.df = df
        self.label_col = label_col
        self.time_cols = [
            col for col in df.columns if col != label_col
        ]  # Todas as colunas exceto a de rótulos
        self.labels = df[label_col].unique()
        self.df_time = self.df.drop(columns=[label_col])
        self.label_names = label_names

    def plot_samples(self):
        """Exibe e retorna a figura com o número total de amostras por classe (nome da classe ao invés do ID)."""
        df_ = (
            self.df.copy()
        )  # Criar uma cópia do DataFrame para evitar modificações diretas no original
        label_col_ = self.label_col
        label_names_ = self.label_names

        # Criar um dicionário de mapeamento de IDs para nomes de atividades
        label_to_activity_ = {i: name for i, name in enumerate(label_names_)}

        # Substituir IDs pelos nomes das atividades na coluna de rótulos
        df_[label_col_] = df_[label_col_].map(label_to_activity_)

        # Verificar se houve falhas no mapeamento
        if df_[label_col_].isnull().any():
            missing_ids = df_[label_col_].isnull()
            print(f"IDs não mapeados: {df_.loc[missing_ids, label_col_].unique()}")
            return None

        # Contar o número de amostras por nome de atividade
        counts_ = df_[label_col_].value_counts()

        # Verificar se há dados
        if counts_.empty:
            print("Nenhum dado disponível para plotagem.")
            return None

        # Criar o gráfico
        fig1, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=counts_.index, y=counts_.values, ax=ax)
        ax.set_title("Número de amostras por classe")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Contagem")

        plt.close()
        return fig1

    def tsne_plot(self, n_components=2):
        """Aplica t-SNE e retorna a figura com os dados reduzidos."""
        tsne = TSNE(n_components=n_components, random_state=42)
        X = self.df[self.time_cols].values
        X_tsne = tsne.fit_transform(X)
        df_tsne = pd.DataFrame(X_tsne, columns=[f"TSNE_{i+1}" for i in range(n_components)])
        df_tsne[self.label_col] = self.df[self.label_col].values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_tsne, x="TSNE_1", y="TSNE_2", hue=self.label_col, palette="Set1", ax=ax
        )
        ax.set_title("t-SNE das séries temporais")
        # plt.close()
        return fig

    def plot_i_samples(
        self,
        activity_names=["sit", "stand", "walk", "stair up", "stair down", "run"],
        n_samples=3,
        reshape=False,
    ):
        df = self.df.copy()

        if "label" not in df.columns:
            raise ValueError("The DataFrame does not contain a 'labels' column.")

        # Create a mapping of activity names to their corresponding label IDs
        label_map = {
            name: i for i, name in enumerate(activity_names)
        }  # Assuming IDs are 1-indexed
        n_activities = len(activity_names)
        samples_per_activity = []

        for activity, label_id in label_map.items():
            indices = df.index[df["label"] == label_id].tolist()
            if not indices:  # Check if there are indices for the label
                print(f"No samples found for label: {activity}")
                continue

            selected_real = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
            samples = self.df_time.loc[selected_real].values

            if reshape:
                real_samples = samples.reshape(-1, 60, 6)

            samples_per_activity.append(samples)

        # Plot
        fig, axs = plt.subplots(
            n_activities, n_samples, figsize=(12, n_activities * 3), squeeze=False
        )

        for col, (activity, samples) in enumerate(zip(activity_names, samples_per_activity)):
            for row in range(min(n_samples, len(samples))):
                if len(samples) > 0:  # Check if there are samples using length
                    axs[col, row].imshow(samples[row].reshape(60, 6), aspect="auto")
                    axs[col, row].set_title(f"{activity} {row + 1}")

                    # Remove axis ticks and labels
                    axs[col, row].axis("off")  # Hide the entire axis
                else:
                    axs[col, row].axis("off")  # Hide the axis if no samples

        # plt.tight_layout()
        plt.close()
        return fig

    def plot_metrics_comparison(self, metrics, thresholds):
        """
        Args:
        - metrics (dict): Um dicionário contendo os nomes das métricas como chaves e seus respectivos valores.
        - thresholds (dict): Um dicionário com os limites de valores "ok" para cada métrica.

        Retorna:
        - None: Exibe os subplots para comparação.
        """
        # Filtrar apenas métricas com valores numéricos (remover None)
        numeric_metrics = {k: v for k, v in metrics.items() if v is not None}

        # Número de métricas a serem plotadas
        n_metrics = len(numeric_metrics)

        if n_metrics == 0:
            print("Nenhuma métrica numérica disponível para plotar.")
            return

        # Preparar os dados para o gráfico
        metric_names = list(numeric_metrics.keys())
        metric_values = list(numeric_metrics.values())

        # Criar subplots
        fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 4 * n_metrics))  # Subplots verticais

        # Se houver apenas uma métrica, garantir que axes seja uma lista
        if n_metrics == 1:
            axes = [axes]

        # Plotar as métricas em subplots individuais
        for i, (ax, metric) in enumerate(zip(axes, numeric_metrics)):
            value = numeric_metrics[metric]
            # Obter limites de "ok" e "ruim"
            threshold = thresholds.get(metric, (None, None))

            # Plotar o valor da métrica como linha
            ax.plot([metric], [value], marker="o", color="black", label=f"{metric}: {value:.3f}")

            # Adicionar linha vermelha ou verde de acordo com os thresholds
            if threshold[0] is not None and value < threshold[0]:
                ax.axhline(threshold[0], color="red", linestyle="--", label="Limite Inferior")
            if threshold[1] is not None and value >= threshold[0]:
                ax.axhline(threshold[1], color="green", linestyle="--", label="Limite Superior")

            # Ajustar título, rótulos e limites
            ax.set_title(f"Métrica: {metric}")
            ax.set_ylim(
                min(value - 0.5, threshold[0] if threshold[0] is not None else value - 0.5),
                max(value + 0.5, threshold[1] if threshold[1] is not None else value + 0.5),
            )
            # Remover rótulos do eixo X (não necessário para este gráfico)
            ax.set_xticks([])
            ax.legend(loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.7)

        # Ajustar layout
        plt.tight_layout()
        plt.close()
        return fig

    def plot_distribution(self):
        """Plota e retorna a figura da distribuição dos dados por classe."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=self.df, x=self.label_col, ax=ax)
        ax.set_title("Distribuição das classes")
        plt.close()
        return fig

    def histogram_density(self):
        """Exibe um histograma e gráfico de densidade das séries temporais."""
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df_time.values.flatten(), kde=True)
        plt.title("Histograma e Gráfico de Densidade das Séries Temporais")
        # plt.show()

    def plot_acf_pacf_all(self):
        fig = plt.figure(figsize=(12, 6))

        # ACF
        plt.subplot(1, 2, 1)
        for col in self.df.columns:
            if col != "label":
                plot_acf(self.df[col], lags=40, alpha=0.05, label=col)
        plt.title("ACF de Todas as Labels")
        plt.legend()

        # PACF
        plt.subplot(1, 2, 2)
        for col in self.df.columns:
            if col != "label":
                plot_pacf(self.df[col], lags=40, alpha=0.05, label=col)
        plt.title("PACF de Todas as Labels")
        plt.legend()

        plt.tight_layout()
        # plt.show()
        return fig

    def plot_autocorrelation(self):
        """Plota a autocorrelação para cada label no DataFrame em subplots."""
        n_labels = len(self.labels)
        n_time_cols = len(self.time_cols)

        # Define o número de subplots
        fig, axs = plt.subplots(n_labels, n_time_cols, figsize=(15, 4 * n_labels))

        for i, label in enumerate(self.labels):
            for j, col in enumerate(self.time_cols):
                label_data = self.df[self.df[self.label_col] == label][col]
                plot_acf(label_data, lags=40, ax=axs[i, j])
                axs[i, j].set_title(f"Autocorrelação - Label: {label}, Coluna: {col}")
                axs[i, j].set_xlabel("Lags")
                axs[i, j].set_ylabel("Autocorrelação")

        plt.tight_layout()
        # plt.show()

    import numpy as np
    import matplotlib.pyplot as plt

    def plot_random_sensor_samples_single_dataset(
        self, df, activity_names, num_samples=3, sensor="accel"
    ):
        """
        Plots random samples of accelerometer data for specific activities in a single dataset.

        Parameters:
        df (DataFrame): The DataFrame containing the dataset to be plotted.
        activity_names (list of str): List of activity names to be plotted.
        num_samples (int): Number of random samples to be plotted for each activity.
        """
        # Ensure num_samples is an integer
        if not isinstance(num_samples, int):
            raise ValueError("num_samples must be an integer.")

        # Ensure num_samples is valid
        if num_samples <= 0:
            print("Number of samples must be greater than zero.")
            return
        # Create subplots for num_samples rows and 1 column
        fig, axes = plt.subplots(
            ncols=num_samples, nrows=6, figsize=(5 * num_samples, 5 * 6), sharex=True, sharey=True
        )
        fig.suptitle(f"{sensor} Data Samples", fontsize=16)

        for act_id, activity_name in enumerate(activity_names):

            # Filter data for the specific activity
            activity_data = df[df["label"] == act_id]

            if activity_data.empty:
                print(f"No data available for activity: {activity_name}")
                continue

            # Identify the columns for sensor data
            accel_columns_x = [col for col in df.columns if col.startswith(f"{sensor}-x")]
            accel_columns_y = [col for col in df.columns if col.startswith(f"{sensor}-y")]
            accel_columns_z = [col for col in df.columns if col.startswith(f"{sensor}-z")]

            # Select random sample indices
            if len(activity_data) < num_samples:
                print(
                    f"Not enough samples available for activity: {activity_name}. Requested: {num_samples}, Available: {len(activity_data)}"
                )
                continue

            sample_indices = np.random.choice(activity_data.index, num_samples, replace=False)

            for j, sample_index in enumerate(sample_indices):
                sample_data = activity_data.loc[sample_index]

                # Extract accelerometer data
                accel_x_data = sample_data[accel_columns_x].values
                accel_y_data = sample_data[accel_columns_y].values
                accel_z_data = sample_data[accel_columns_z].values

                # Plot data for each sample
                ax = axes[act_id, j]
                ax.plot(accel_x_data, color="r", alpha=0.7, label=f"{sensor} X")
                ax.plot(accel_y_data, color="g", alpha=0.7, label=f"{sensor} Y")
                ax.plot(accel_z_data, color="b", alpha=0.7, label=f"{sensor} Z")

                ax.set_title(f"{activity_name} {j + 1}")
                ax.set_xlabel("Time")
                # ax.set_ylabel('Accelera'Acceltion')
                ax.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.97])
        # plt.close()
        return fig

    def plot_correlation_matrix(self):
        """Plota e retorna a figura da matriz de correlação das características temporais."""
        corr_matrix = self.df[self.time_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Matriz de Correlação das Características")

        return fig

    def plotly_distribution(self):
        """Retorna a figura da distribuição das classes usando Plotly."""
        fig = px.histogram(self.df, x=self.label_col, title="Distribuição de Classes")
        return fig

    def apply_to_multiple_samples(self, label_map, n_samples=3, technique="line_plot"):
        """
        Aplica uma técnica a múltiplas amostras por classe.

        label_map: Dicionário mapeando nomes das atividades para IDs dos rótulos.
        n_samples: Número de amostras a serem selecionadas por classe.
        technique: Técnica de visualização a ser aplicada ('line_plot', 'lag_plot', etc.)
        """
        for activity, label_id in label_map.items():
            indices = self.df.index[self.df[self.label_col] == label_id].tolist()
            if not indices:  # Verifica se há amostras para o rótulo
                print(f"Nenhuma amostra encontrada para a atividade: {activity}")
                continue

            selected_real = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
            samples = self.df_time.loc[selected_real].values

            print(
                f"Aplicando técnica {technique} para {n_samples} amostras da atividade: {activity}"
            )

            for sample_idx in selected_real:
                if technique == "line_plot":
                    self.line_plot(sample_idx)
                elif technique == "lag_plot":
                    self.lag_plot(sample_idx)
                elif technique == "autocorrelation_plot":
                    self.autocorrelation_plot(sample_idx)
                elif technique == "decompose_time_series":
                    self.decompose_time_series(sample_idx)
                elif technique == "rolling_mean":
                    self.rolling_mean(sample_idx)
                elif technique == "fourier_transform":
                    self.fourier_transform(sample_idx)
                else:
                    print(f"Técnica {technique} não suportada.")
