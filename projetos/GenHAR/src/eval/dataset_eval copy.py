import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
import plotly.express as px
import numpy as np

class TimeSeriesDatasetEvaluator:
    def __init__(self, df, label_col):
        """
        Classe para avaliar o dataset de séries temporais.
        
        df: DataFrame que contém os dados de séries temporais.
        label_col: Nome da coluna que contém os rótulos.
        """
        self.df = df
        self.label_col = label_col
        self.time_cols = [col for col in df.columns if col != label_col]  # Todas as colunas exceto a de rótulos
        self.labels = df[label_col].unique()
        self.df_time = self.df.drop(columns=[label_col])

    def apply_to_multiple_samples(self, label_map, n_samples=3, technique='line_plot'):
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
            
            print(f"Aplicando técnica {technique} para {n_samples} amostras da atividade: {activity}")
            
            for sample_idx in selected_real:
                if technique == 'line_plot':
                    self.line_plot(sample_idx)
                elif technique == 'lag_plot':
                    self.lag_plot(sample_idx)
                elif technique == 'autocorrelation_plot':
                    self.autocorrelation_plot(sample_idx)
                elif technique == 'decompose_time_series':
                    self.decompose_time_series(sample_idx)
                elif technique == 'rolling_mean':
                    self.rolling_mean(sample_idx)
                elif technique == 'fourier_transform':
                    self.fourier_transform(sample_idx)
                else:
                    print(f"Técnica {technique} não suportada.")
    
    def line_plot(self, sample_idx):
        """Cria um gráfico de linha para uma amostra de séries temporais."""
        sample = self.df_time.iloc[sample_idx]
        plt.figure(figsize=(12, 6))
        plt.plot(sample)
        plt.title(f"Gráfico de Linha para a Amostra {sample_idx}")
        plt.xlabel("Tempo")
        plt.ylabel("Valor")
        plt.show()
    
    def lag_plot(self, sample_idx, lag=1):
        """Cria um gráfico de defasagem (lag plot) para uma amostra de séries temporais."""
        sample = self.df_time.iloc[sample_idx]
        pd.plotting.lag_plot(sample, lag=lag)
        plt.title(f"Gráfico de Defasagem (Lag Plot) com Lag={lag} para Amostra {sample_idx}")
        plt.show()

    def autocorrelation_plot(self, sample_idx):
        """Cria um gráfico de autocorrelação (ACF) para uma amostra de séries temporais."""
        from pandas.plotting import autocorrelation_plot
        sample = self.df_time.iloc[sample_idx]
        autocorrelation_plot(sample)
        plt.title(f"Gráfico de Autocorrelação para Amostra {sample_idx}")
        plt.show()
    
    def decompose_time_series(self, sample_idx, model='additive', period=12):
        """Decompõe uma série temporal em tendência, sazonalidade e ruído."""
        sample = self.df_time.iloc[sample_idx]
        result = seasonal_decompose(sample, model=model, period=period)
        result.plot()
        plt.suptitle(f"Decomposição da Série Temporal (Amostra {sample_idx})", y=1.02)
        plt.show()

    def rolling_mean(self, sample_idx, window=12):
        """Exibe a média móvel de uma série temporal."""
        sample = self.df_time.iloc[sample_idx]
        rolling_mean = sample.rolling(window=window).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(sample, label='Original')
        plt.plot(rolling_mean, label='Média Móvel')
        plt.title(f"Média Móvel (Window={window}) para Amostra {sample_idx}")
        plt.legend()
        plt.show()

    def fourier_transform(self, sample_idx):
        """Aplica a Transformada de Fourier em uma série temporal e plota as frequências."""
        sample = self.df_time.iloc[sample_idx]
        fft_vals = fft(sample)
        fft_freqs = np.fft.fftfreq(len(fft_vals))
        
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freqs, np.abs(fft_vals))
        plt.title(f"Transformada de Fourier para Amostra {sample_idx}")
        plt.xlabel("Frequência")
        plt.ylabel("Magnitude")
        plt.show()
