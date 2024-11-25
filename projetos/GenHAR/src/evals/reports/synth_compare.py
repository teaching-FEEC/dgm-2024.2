import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class SyntheticComparator:
    def __init__(self, df_train, df_synths, synt_df_names, label_col='label', activity_names=None):
        """
        Inicializa a classe com o dataset real e uma lista de datasets sintéticos.
        """
        # Verificar se a coluna 'label' existe nos dados reais
        if label_col not in df_train.columns:
            raise KeyError(f"A coluna '{label_col}' não existe no dataframe real.")
        
        self.df_real = df_train
        self.label_col = label_col  # A coluna de rótulo
        self.feature_cols = self.df_real.columns.drop(label_col)  # Colunas de características
        
        # Verificar se a coluna 'label' existe em todos os datasets sintéticos
        for i, df_synt in enumerate(df_synths):
            if label_col not in df_synt.columns:
                raise KeyError(f"A coluna '{label_col}' não existe no dataset sintético {synt_df_names[i]}.")
        
        # Armazenar os datasets sintéticos sem remover a coluna de rótulo
        self.df_synthetics_list_df = df_synths
        self.synt_df_names = synt_df_names
        self.activity_names = {i: name for i, name in enumerate(activity_names)} if activity_names is not None else {}

    def compare_class_distribution(self):
        """
        Compara e retorna a figura da distribuição das classes entre datasets real e sintético.
        """
        # Distribuição das classes no dataset real
        real_counts = self.df_real[self.label_col].value_counts().sort_index()
        
        # Lista para armazenar as distribuições sintéticas
        synthetic_counts_list = []
        
        # Unir todas as classes dos datasets sintéticos
        all_classes_set = set(real_counts.index)
        
        for df_synt in self.df_synthetics_list_df:
            all_classes_set.update(df_synt[self.label_col].unique())
        
        all_classes = pd.Index(sorted(all_classes_set))  # Ordenar as classes
        
        # Adicionando as contagens para o dataset real
        df_counts = pd.DataFrame({
            'Class': all_classes,
            'Real': real_counts.reindex(all_classes, fill_value=0).values  # Usando fill_value em vez de fillvalue
        })
        
        # Adicionando as contagens sintéticas ao DataFrame
        for i, (df_synt, synt_name) in enumerate(zip(self.df_synthetics_list_df, self.synt_df_names)):
            synt_counts = df_synt[self.label_col].value_counts().sort_index()
            df_counts[synt_name] = synt_counts.reindex(all_classes, fill_value=0).values  # Usando o nome do dataset sintético
        
        # Transformando o DataFrame para o formato 'long' para o Seaborn
        df_counts = df_counts.melt(id_vars='Class', var_name='Dataset', value_name='Count')
        
        # Mapear IDs das classes para os nomes das atividades
        if self.activity_names:
            df_counts['Class'] = df_counts['Class'].map(lambda x: self.activity_names.get(x, x))
        
        # Plotando a distribuição das classes
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=df_counts, x='Class', y='Count', hue='Dataset', ax=ax)
        ax.set_title("Distribuição de Classes: Real vs Sintético")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Contagem")
        ax.legend(title='Dataset')

        plt.xticks(rotation=45)  # Rotacionar rótulos do eixo x para melhorar a visibilidade
        plt.tight_layout()  # Ajustar o layout
        return fig

    def visualize_tsne_unlabeled(self, perplexity=10, markersize=20, alpha=0.5):
        """
        Visualiza a t-SNE não supervisionada (sem rótulos) para o dataset real e os datasets sintéticos.

        :param perplexity: Parâmetro de perplexidade para o t-SNE.
        :param markersize: Tamanho dos pontos no gráfico.
        :param alpha: Transparência dos pontos no gráfico.
        """
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        
        # Padronizar dados reais (sem a coluna de rótulos)
        x_real = self.df_real[self.feature_cols].values
        x_real = StandardScaler().fit_transform(x_real)
        
        # Padronizar dados sintéticos e combinar todos os dados
        synthetic_data_list = []
        labels = ['Real'] * len(x_real)
        
        for i, df_synt in enumerate(self.df_synthetics_list_df):
            # Remover a coluna de rótulos se existir
            x_synt = df_synt[self.feature_cols].values
            x_synt = StandardScaler().fit_transform(x_synt)
            synthetic_data_list.append(x_synt)
            labels += [self.synt_df_names[i]] * len(x_synt)
        
        # Combinar dados reais e sintéticos
        combined_data = np.vstack([x_real] + synthetic_data_list)
        
        # Aplicar t-SNE em todos os dados combinados
        X_all_tsne = tsne.fit_transform(combined_data)
        
        # Criar DataFrame para visualização
        df_combined = pd.DataFrame(X_all_tsne, columns=["TSNE_1", "TSNE_2"])
        df_combined['Dataset'] = labels
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_combined, x="TSNE_1", y="TSNE_2", hue="Dataset", style="Dataset", s=markersize, alpha=alpha, ax=ax)
        ax.set_title("t-SNE Unlabeled: Real vs Sintéticos")
        
        # Ajustando a legenda para não sobrecarregar o gráfico
        plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()  # Ajustar layout para melhorar visualização
        return fig
    
    def visualize_distribution(self):
        """
        Compara a distribuição dos dados reais e sintéticos para vários datasets sintéticos,
        retornando uma única figura com todas as distribuições.
        """
        # Obter os dados reais
        x_real = self.df_real[self.feature_cols].values
        y_real = self.df_real[self.label_col].values
        
        # Certificar que o número de amostras seja o mínimo entre os dois conjuntos
        sample_num = min([1000, len(x_real)])
        
        # Selecionar amostras aleatórias dos dados reais
        idx = np.random.permutation(len(x_real))[:sample_num]
        x_real = x_real[idx]
        
        # Cálculo da média ao longo da série temporal para os dados reais
        prep_data = np.mean(x_real, axis=1)

        # Aplicar MinMaxScaler para escalar os dados reais entre 0 e 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        prep_data_scaled = scaler.fit_transform(prep_data.reshape(-1, 1)).flatten()

        # Plotar a distribuição
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # KDE plot para os dados reais
        sns.kdeplot(prep_data_scaled, color='C0', linewidth=2, label='Real', ax=ax)
        
        # Para cada dataset sintético, calcular a média e gerar o KDE
        for i, df_synt in enumerate(self.df_synthetics_list_df):
            x_synthetic = df_synt[self.feature_cols].values
            
            # Selecionar amostras aleatórias para os dados sintéticos
            x_synthetic = x_synthetic[:sample_num]  # Garantir que não acesse índices fora de alcance
            
            # Cálculo da média ao longo da série temporal para os dados sintéticos
            prep_data_hat = np.mean(x_synthetic, axis=1)
            
            # Escalar os dados sintéticos
            prep_data_hat_scaled = scaler.transform(prep_data_hat.reshape(-1, 1)).flatten()

            # Plotar KDE para o dataset sintético atual
            sns.kdeplot(prep_data_hat_scaled, linewidth=2, linestyle='--', label=f'{self.synt_df_names[i]}', ax=ax)

        # Definindo os limites do eixo X entre 0 e 1 (após escalamento)
        ax.set_xlim(0, 1)

        # Customização do gráfico
        ax.set_xlabel('Valor Escalado')
        ax.set_ylabel('Densidade')
        ax.set_title('Distribuição de Classes: Real vs Sintéticos')
        
        # Remover os spines superiores e diretos
        for pos in ['top', 'right']:
            ax.spines[pos].set_visible(False)

        # Adicionar uma legenda
        ax.legend(title="Dataset")
        plt.close(fig)  # Fechar a figura após salvar
        return fig
    
    def tsne_subplots_by_labels(self):
        """
        Compara os datasets reais e sintéticos para cada label (atividade) usando t-SNE,
        criando subgráficos para cada label de atividade, diferenciando os dados sintéticos por cores.
        """
        unique_labels = self.df_real[self.label_col].unique()
        cols = 2
        rows = (len(unique_labels) + 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 5))
        colors = sns.color_palette("husl", n_colors=len(self.synt_df_names) + 1)

        for i, label in enumerate(unique_labels):
            row = i // cols
            col = i % cols

            # Filtrar dados reais e sintéticos pelo label atual
            real_data = self.df_real[self.df_real[self.label_col] == label][self.feature_cols].values
            real_data_labels = self.df_real[self.df_real[self.label_col] == label][self.label_col].values

            synthetic_data = []
            synthetic_labels = []
            for df_synt in self.df_synthetics_list_df:
                synt_data = df_synt[df_synt[self.label_col] == label][self.feature_cols].values
                synt_labels = df_synt[df_synt[self.label_col] == label][self.label_col].values

                synthetic_data.append(synt_data)
                synthetic_labels.append(synt_labels)

            # Concatenar dados reais e sintéticos para este label
            combined_data = np.concatenate([real_data] + synthetic_data)
            combined_labels = np.concatenate([real_data_labels] + synthetic_labels)

            # Normalizar os dados
            combined_data = StandardScaler().fit_transform(combined_data)

            # Aplicar t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(combined_data)

            # Criar DataFrame para plotagem
            df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])
            df['Source'] = ['Real'] * len(real_data_labels) + sum(
                [[name] * len(labels) for name, labels in zip(self.synt_df_names, synthetic_labels)], []
            )
            df['Label'] = combined_labels

            # Plotar os dados reais
            sns.scatterplot(
                data=df[df['Source'] == 'Real'],
                x='Component 1',
                y='Component 2',
                ax=axs[row, col],
                color=colors[0],
                label='Real'
            )

            # Plotar os dados sintéticos com cores distintas
            for j, name in enumerate(self.synt_df_names):
                sns.scatterplot(
                    data=df[df['Source'] == name],
                    x='Component 1',
                    y='Component 2',
                    ax=axs[row, col],
                    color=colors[j + 1],
                    label=name,
                    marker='X'
                )

            # Adicionar título e legenda
            title = f"Label: {self.activity_names[label]}" if label < len(self.activity_names) else f"Label: {label}"
            axs[row, col].set_title(title)
            axs[row, col].legend()

        # Ajustar layout
        plt.tight_layout()
        return fig

    def plot_random_samples_comparison_by_labels(self, num_samples=3):
        """
        Plots random samples of all sensor data for specific activities across real and multiple synthetic datasets.

        Parameters:
        num_samples (int): Number of random samples to be plotted for each activity.
        """
        # Validar o número de amostras
        if not isinstance(num_samples, int):
            raise ValueError("num_samples must be an integer.")
        if num_samples <= 0:
            print("Number of samples must be greater than zero.")
            return

        # Número de linhas: uma para os dados reais + uma para cada dataset sintético
        total_rows = 1 + len(self.df_synthetics_list_df)
        total_activities = len(self.activity_names)

        # Criar subplots: uma linha por dataset (real + sintéticos) para cada atividade, várias colunas para amostras
        fig, axes = plt.subplots(
            nrows=total_rows * total_activities,
            ncols=num_samples,
            figsize=(5 * num_samples, 5 * total_rows * total_activities),
            sharex=True,
            sharey=False,
        )
        axes = np.atleast_2d(axes)  # Garantir que axes seja uma matriz 2D
        fig.suptitle('Data Samples Comparison: Real vs Synthetic', fontsize=16)

        # Iterar sobre cada atividade
        for act_id, activity_name in enumerate(self.activity_names.values()):  # Use o values() para acessar os nomes
            # Índice base para a linha atual
            base_row = act_id * total_rows

            # Obter dados reais para a atividade
            real_activity_data = self.df_real[self.df_real[self.label_col] == act_id]
            if real_activity_data.empty:
                print(f"No data available for activity in Real dataset: {activity_name}")
                continue

            # Amostrar aleatoriamente dados reais
            real_sample_indices = np.random.choice(real_activity_data.index, min(num_samples, len(real_activity_data)), replace=False)

            # Linha para os dados reais
            for j, real_sample_index in enumerate(real_sample_indices):
                real_sample_data = real_activity_data.loc[real_sample_index]
                ax_real = axes[base_row, j]
                ax_real.plot(real_sample_data[self.feature_cols].values, color='c', alpha=0.7)
                ax_real.set_title(f'Real: {activity_name} {j + 1}')  # Usar o nome da atividade
                if j == 0:
                    ax_real.set_ylabel('Sensor Values')
                ax_real.set_xlabel('Time')

            # Iterar sobre os datasets sintéticos
            for row_offset, (df_synthetic, synt_name) in enumerate(zip(self.df_synthetics_list_df, self.synt_df_names), start=1):
                synthetic_activity_data = df_synthetic[df_synthetic[self.label_col] == act_id]
                if synthetic_activity_data.empty:
                    print(f"No data available for activity in Synthetic dataset ({synt_name}): {activity_name}")
                    continue

                # Amostrar aleatoriamente dados sintéticos
                synthetic_sample_indices = np.random.choice(
                    synthetic_activity_data.index, min(num_samples, len(synthetic_activity_data)), replace=False
                )

                for j, synthetic_sample_index in enumerate(synthetic_sample_indices):
                    synthetic_sample_data = synthetic_activity_data.loc[synthetic_sample_index]
                    ax_synthetic = axes[base_row + row_offset, j]
                    ax_synthetic.plot(synthetic_sample_data[self.feature_cols].values, color='m', alpha=0.7)
                    ax_synthetic.set_title(f'{synt_name}: {activity_name} {j + 1}')  # Usar o nome da atividade
                    if j == 0:
                        ax_synthetic.set_ylabel('Sensor Values')
                    ax_synthetic.set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig
        
    def plot_random_samples_comparison_by_labels(self, num_samples=3):
        """
        Plots random samples of all sensor data for specific activities across real and multiple synthetic datasets.

        Parameters:
        num_samples (int): Number of random samples to be plotted for each activity.
        """
        # Validar o número de amostras
        if not isinstance(num_samples, int):
            raise ValueError("num_samples must be an integer.")
        if num_samples <= 0:
            print("Number of samples must be greater than zero.")
            return

        # Número de linhas: uma para os dados reais + uma para cada dataset sintético
        total_rows = 1 + len(self.df_synthetics_list_df)
        total_activities = len(self.activity_names)

        # Criar subplots: uma linha por dataset (real + sintéticos) para cada atividade, várias colunas para amostras
        fig, axes = plt.subplots(
            nrows=total_rows * total_activities,
            ncols=num_samples,
            figsize=(5 * num_samples, 5 * total_rows * total_activities),
            sharex=True,
            sharey=False,
        )
        axes = np.atleast_2d(axes)  # Garantir que axes seja uma matriz 2D
        fig.suptitle('Data Samples Comparison: Real vs Synthetic', fontsize=16)

        # Iterar sobre cada atividade
        for act_id, activity_name in enumerate(self.activity_names.values()):  # Use o values() para acessar os nomes
            # Índice base para a linha atual
            base_row = act_id * total_rows

            # Obter dados reais para a atividade
            real_activity_data = self.df_real[self.df_real[self.label_col] == act_id]
            if real_activity_data.empty:
                print(f"No data available for activity in Real dataset: {activity_name}")
                continue

            # Amostrar aleatoriamente dados reais
            real_sample_indices = np.random.choice(real_activity_data.index, min(num_samples, len(real_activity_data)), replace=False)

            # Linha para os dados reais
            for j, real_sample_index in enumerate(real_sample_indices):
                real_sample_data = real_activity_data.loc[real_sample_index]

                
                ax_real = axes[base_row, j]
                ax_real.plot(real_sample_data[self.feature_cols].values, color='c', alpha=0.7)
                ax_real.set_title(f'Real: {activity_name} {j + 1}')  # Usar o nome da atividade
                if j == 0:
                    ax_real.set_ylabel('Sensor Values')
                ax_real.set_xlabel('Time')

            # Iterar sobre os datasets sintéticos
            for row_offset, (df_synthetic, synt_name) in enumerate(zip(self.df_synthetics_list_df, self.synt_df_names), start=1):
                synthetic_activity_data = df_synthetic[df_synthetic[self.label_col] == act_id]
                if synthetic_activity_data.empty:
                    print(f"No data available for activity in Synthetic dataset ({synt_name}): {activity_name}")
                    continue

                # Amostrar aleatoriamente dados sintéticos
                synthetic_sample_indices = np.random.choice(
                    synthetic_activity_data.index, min(num_samples, len(synthetic_activity_data)), replace=False
                )

                for j, synthetic_sample_index in enumerate(synthetic_sample_indices):
                    synthetic_sample_data = synthetic_activity_data.loc[synthetic_sample_index]
                    ax_synthetic = axes[base_row + row_offset, j]
                    ax_synthetic.plot(synthetic_sample_data[self.feature_cols].values, color='m', alpha=0.7)
                    ax_synthetic.set_title(f'{synt_name}: {activity_name} {j + 1}')  # Usar o nome da atividade
                    if j == 0:
                        ax_synthetic.set_ylabel('Sensor Values')
                    ax_synthetic.set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig






    def plot_random_sensor_samples_comparison(self, num_samples=3, sensor="accel"):
        """
        Plots random samples of accelerometer data for specific activities in both real and synthetic datasets.

        Parameters:
        num_samples (int): Number of random samples to be plotted for each activity.
        sensor (str): Type of sensor data to plot (e.g., 'accel', 'gyro').
        """
        # Ensure num_samples is an integer
        if not isinstance(num_samples, int):
            raise ValueError("num_samples must be an integer.")
        
        # Ensure num_samples is valid
        if num_samples <= 0:
            print("Number of samples must be greater than zero.")
            return
        figs=[]
        for act_id, activity_name in enumerate(self.activity_names):
            # Filter data for the specific activity in the real dataset
            real_activity_data = self.df_real[self.df_real[self.label_col] == act_id]

            if real_activity_data.empty:
                print(f"No data available for activity in Real dataset: {activity_name}")
                continue

            # Combine synthetic data from all synthetic datasets in the list
            synthetic_activity_data = pd.concat(
                [df[df[self.label_col] == act_id] for df in self.df_synthetics_list_df], ignore_index=True
            )

            if synthetic_activity_data.empty:
                print(f"No data available for activity in Synthetic dataset: {activity_name}")
                continue
            
            # Identify the columns for sensor data
            accel_columns_x = [col for col in self.df_real.columns if col.startswith(f'{sensor}-x')]
            accel_columns_y = [col for col in self.df_real.columns if col.startswith(f'{sensor}-y')]
            accel_columns_z = [col for col in self.df_real.columns if col.startswith(f'{sensor}-z')]

            # Select random sample indices for both datasets
            real_sample_indices = np.random.choice(real_activity_data.index, min(num_samples, len(real_activity_data)), replace=False)
            synthetic_sample_indices = np.random.choice(synthetic_activity_data.index, min(num_samples, len(synthetic_activity_data)), replace=False)

            # Create a new figure for each activity label
            fig, axes = plt.subplots(nrows=len(self.df_synthetics_list_df) + 1, ncols=num_samples, 
                                     figsize=(5 * num_samples, 5 * (len(self.df_synthetics_list_df) + 1)),
                                     sharex=True, sharey='row')
            
            fig.suptitle(f'{sensor} Data Samples Comparison for Activity: { self.activity_names[activity_name]}', fontsize=16)

            # Plot real samples (Row 1)
            for j, real_sample_index in enumerate(real_sample_indices):
                real_sample_data = real_activity_data.loc[real_sample_index]

                # Extract accelerometer data for real sample
                accel_x_data_real = real_sample_data[accel_columns_x].values
                accel_y_data_real = real_sample_data[accel_columns_y].values
                accel_z_data_real = real_sample_data[accel_columns_z].values

                # Plot data for real sample
                ax_real = axes[0, j]
                ax_real.plot(accel_x_data_real, color='r', alpha=0.7, label=f'{sensor} X (Real)')
                ax_real.plot(accel_y_data_real, color='g', alpha=0.7, label=f'{sensor} Y (Real)')
                ax_real.plot(accel_z_data_real, color='b', alpha=0.7, label=f'{sensor} Z (Real)')
                ax_real.set_title(f'Real Sample {j + 1}')
                ax_real.set_xlabel('Time')
                ax_real.legend()

            # Plot synthetic samples (Rows 2 to n)
            for row, synthetic_data in enumerate(self.df_synthetics_list_df):
                synthetic_activity_data = synthetic_data[synthetic_data[self.label_col] == act_id]
                synthetic_sample_indices = np.random.choice(synthetic_activity_data.index, min(num_samples, len(synthetic_activity_data)), replace=False)
                
                for j, synthetic_sample_index in enumerate(synthetic_sample_indices):
                    synthetic_sample_data = synthetic_activity_data.loc[synthetic_sample_index]

                    # Extract accelerometer data for synthetic sample
                    accel_x_data_synthetic = synthetic_sample_data[accel_columns_x].values
                    accel_y_data_synthetic = synthetic_sample_data[accel_columns_y].values
                    accel_z_data_synthetic = synthetic_sample_data[accel_columns_z].values

                    # Plot data for synthetic sample
                    ax_synthetic = axes[row + 1, j]
                    ax_synthetic.plot(accel_x_data_synthetic, color='r', alpha=0.7, label=f'{sensor} X (Synthetic)')
                    ax_synthetic.plot(accel_y_data_synthetic, color='g', alpha=0.7, label=f'{sensor} Y (Synthetic)')
                    ax_synthetic.plot(accel_z_data_synthetic, color='b', alpha=0.7, label=f'{sensor} Z (Synthetic)')
                    ax_synthetic.set_title(f' {self.synt_df_names[row]}{j + 1} ')
                    ax_synthetic.set_xlabel('Time')
                    ax_synthetic.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            figs.append(fig)
        return figs
