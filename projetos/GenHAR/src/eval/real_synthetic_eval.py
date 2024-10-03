import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
import plotly.express as px

class RealSyntheticComparator:
    def __init__(self, df_real, df_synthetic, label_col,activity_names):
        """
        Classe para comparar o dataset real e o dataset sintético.
        
        df_real: DataFrame contendo o dataset real.
        df_synthetic: DataFrame contendo o dataset sintético.
        label_col: Nome da coluna que contém os rótulos.
        """
        self.df_real = df_real
        self.df_synthetic = df_synthetic
        self.label_col = label_col
        self.time_cols = [col for col in df_real.columns if col != label_col]  # Todas as colunas exceto a de rótulos
        self.labels = df_real[label_col].unique()
        self.activity_names=activity_names
        
        # Extraindo apenas as colunas temporais
        self.df_real_time = self.df_real.drop(columns=[label_col])
        self.df_synthetic_time = self.df_synthetic.drop(columns=[label_col])

    def compare_class_distribution(self):
        """Compara e retorna a figura da distribuição das classes entre datasets real e sintético."""
        real_counts = self.df_real[self.label_col].value_counts().sort_index()
        synthetic_counts = self.df_synthetic[self.label_col].value_counts().sort_index()
        
        # Create DataFrame for plotting
        df_counts = pd.DataFrame({
            'Real': real_counts,
            'Sintético': synthetic_counts
        }).reset_index().melt(id_vars='index', var_name='Dataset', value_name='Count')

        # Map label IDs to activity names
        df_counts['Class'] = df_counts['index'].map(lambda x: self.activity_names[x])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_counts, x='Class', y='Count', hue='Dataset', ax=ax)
        ax.set_title("Distribuição de Classes: Real vs Sintético")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Contagem")
        ax.legend(title='Dataset')

        plt.xticks(rotation=45)  # Optional: Rotate x labels for better visibility
        return fig

    def compare_tsne(self, n_components=2):
        """Aplica t-SNE em ambos os datasets e retorna a figura de comparação."""
        tsne = TSNE(n_components=n_components, random_state=42)
        
        # Preparando dados
        X_real = StandardScaler().fit_transform(self.df_real[self.time_cols].values)
        X_synthetic = StandardScaler().fit_transform(self.df_synthetic[self.time_cols].values)
        
        # t-SNE
        X_real_tsne = tsne.fit_transform(X_real)
        X_synthetic_tsne = tsne.fit_transform(X_synthetic)
        
        df_real_tsne = pd.DataFrame(X_real_tsne, columns=[f"TSNE_{i+1}" for i in range(n_components)])
        df_real_tsne[self.label_col] = self.df_real[self.label_col].values
        df_real_tsne['Dataset'] = 'Real'
        
        df_synthetic_tsne = pd.DataFrame(X_synthetic_tsne, columns=[f"TSNE_{i+1}" for i in range(n_components)])
        df_synthetic_tsne[self.label_col] = self.df_synthetic[self.label_col].values
        df_synthetic_tsne['Dataset'] = 'Sintético'
        
        df_combined = pd.concat([df_real_tsne, df_synthetic_tsne])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_combined, x="TSNE_1", y="TSNE_2", hue=self.label_col, style='Dataset', palette="Set1", ax=ax)
        ax.set_title("t-SNE: Real vs Sintético")
        
        return fig

    def compare_images(self, activity_names=['sit', 'stand', 'walk', 'stair up', 'stair down', 'run'], n_samples=3, reshape=False):
        """
        Compara amostras aleatórias do dataset real e sintético para cada atividade.
        """
        label_map = {name: i for i, name in enumerate(activity_names)}  # Mapeamento de nome de atividades para IDs
        n_activities = len(activity_names)
        
        samples_per_activity_real = []
        samples_per_activity_synthetic = []

        for activity, label_id in label_map.items():
            # Amostras do dataset real
            real_indices = self.df_real.index[self.df_real[self.label_col] == label_id].tolist()
            selected_real = np.random.choice(real_indices, min(n_samples, len(real_indices)), replace=False)
            real_samples = self.df_real_time.loc[selected_real].values

            # Amostras do dataset sintético
            synthetic_indices = self.df_synthetic.index[self.df_synthetic[self.label_col] == label_id].tolist()
            selected_synthetic = np.random.choice(synthetic_indices, min(n_samples, len(synthetic_indices)), replace=False)
            synthetic_samples = self.df_synthetic_time.loc[selected_synthetic].values

            if reshape:
                real_samples = real_samples.reshape(-1, 60, 6)
                synthetic_samples = synthetic_samples.reshape(-1, 60, 6)

            samples_per_activity_real.append(real_samples)
            samples_per_activity_synthetic.append(synthetic_samples)

        # Plotting
        fig, axs = plt.subplots(n_activities, n_samples * 2, figsize=(15, n_activities * 3), squeeze=False)
        
        for col, (activity, real_samples, synthetic_samples) in enumerate(zip(activity_names, samples_per_activity_real, samples_per_activity_synthetic)):
            for row in range(min(n_samples, len(real_samples))):
                if len(real_samples) > 0:
                    try:
                        axs[col, row].imshow(real_samples[row].reshape(60, 6), aspect='auto')
                        axs[col, row].set_title(f'Real {activity} {row + 1}')
                        axs[col, row + n_samples].imshow(synthetic_samples[row].reshape(60, 6), aspect='auto')
                        axs[col, row + n_samples].set_title(f'Sintético {activity} {row + 1}')
                        axs[col, row].axis('off')
                        axs[col, row + n_samples].axis('off')
                    except:
                        axs[col, row].axis('off')
                        axs[col, row + n_samples].axis('off')

                else:
                    axs[col, row].axis('off')
                    axs[col, row + n_samples].axis('off')

        plt.tight_layout()
        return fig

    def compare_correlation_matrices(self):
        """Compara as matrizes de correlação entre os datasets real e sintético."""
        corr_real = self.df_real[self.time_cols].corr()
        corr_synthetic = self.df_synthetic[self.time_cols].corr()

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(corr_real, annot=False, cmap='coolwarm', ax=axs[0])
        axs[0].set_title("Matriz de Correlação - Real")
        
        sns.heatmap(corr_synthetic, annot=False, cmap='coolwarm', ax=axs[1])
        axs[1].set_title("Matriz de Correlação - Sintético")
        
        return fig
    

    def visualize_tsne_unlabeled(self, perplexity=10, markersize=20, alpha=0.5):
            """
            Visualiza a t-SNE não supervisionada (sem rótulos) para os datasets real e sintético.
            """
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            
            # Padronizar ambos os datasets
            X_real = StandardScaler().fit_transform(self.df_real[self.time_cols].values)
            X_synthetic = StandardScaler().fit_transform(self.df_synthetic[self.time_cols].values)
            
            # Aplicar t-SNE em ambos
            X_real_tsne = tsne.fit_transform(X_real)
            X_synthetic_tsne = tsne.fit_transform(X_synthetic)
            
            # Criar DataFrames para visualização
            df_real_tsne = pd.DataFrame(X_real_tsne, columns=["TSNE_1", "TSNE_2"])
            df_real_tsne["Dataset"] = "Real"
            
            df_synthetic_tsne = pd.DataFrame(X_synthetic_tsne, columns=["TSNE_1", "TSNE_2"])
            df_synthetic_tsne["Dataset"] = "Sintético"
            
            df_combined = pd.concat([df_real_tsne, df_synthetic_tsne])
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_combined, x="TSNE_1", y="TSNE_2", hue="Dataset", style="Dataset", s=markersize, alpha=alpha, ax=ax)
            ax.set_title("t-SNE Unlabeled: Real vs Sintético")
            
            return fig
    
    def tsne_subplots_by_labels(self, activity_names=['sit', 'stand', 'walk', 'stair up', 'stair down', 'run']):
        # Reshape if 3D
        x_real = self.df_real[self.time_cols].values
        x_synthetic = self.df_synthetic[self.time_cols].values
        y_real = self.df_real[self.label_col].values
        y_synthetic = self.df_synthetic[self.label_col].values
        
        if x_real.ndim == 3:
            x_real = x_real.reshape(x_real.shape[0], -1)
        if x_synthetic.ndim == 3:
            x_synthetic = x_synthetic.reshape(x_synthetic.shape[0], -1)

        # Combine data for t-SNE
        combined_data = np.concatenate((x_real, x_synthetic))
        combined_labels = np.concatenate((y_real, y_synthetic))

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(combined_data)

        # Create DataFrame for plotting
        df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])
        df['Label'] = combined_labels
        df['Source'] = ['Real'] * len(y_real) + ['Generated'] * len(y_synthetic)

        unique_labels = np.unique(combined_labels)
        
        # Create subplots
        cols = 2
        rows = (len(unique_labels) + 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 5))

        for i, label in enumerate(unique_labels):
            row = i // cols
            col = i % cols
            
            # Filter data for the specific label and source
            label_data = df[df['Label'] == label]
            label_data_real = label_data[label_data['Source'] == 'Real']
            label_data_gen = label_data[label_data['Source'] == 'Generated']

            # Scatter plots for real and generated data
            sns.scatterplot(data=label_data_real, x='Component 1', y='Component 2', ax=axs[row, col], color='blue', label='Real')
            sns.scatterplot(data=label_data_gen, x='Component 1', y='Component 2', ax=axs[row, col], color='red', label='Generated', marker='X')

            axs[row, col].set_title(f"Label: {activity_names[label]}")  # Map label ID to activity name
            axs[row, col].legend()

        plt.tight_layout()
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

        # Create subplots for num_samples rows and 2 columns (one for each dataset)
        fig, axes = plt.subplots(nrows=len(self.activity_names), ncols=num_samples * 2, 
                                figsize=(5 * num_samples * 2, 5 * len(self.activity_names)), 
                                sharex=True, sharey=True)
        fig.suptitle(f'{sensor} Data Samples Comparison: Real vs Synthetic', fontsize=16)

        for act_id, activity_name in enumerate(self.activity_names):
            # Filter data for the specific activity in both datasets
            real_activity_data = self.df_real[self.df_real[self.label_col] == act_id]
            synthetic_activity_data = self.df_synthetic[self.df_synthetic[self.label_col] == act_id]

            if real_activity_data.empty:
                print(f"No data available for activity in Real dataset: {activity_name}")
                continue
            
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

            for j, real_sample_index in enumerate(real_sample_indices):
                real_sample_data = real_activity_data.loc[real_sample_index]

                # Extract accelerometer data for real sample
                accel_x_data_real = real_sample_data[accel_columns_x].values
                accel_y_data_real = real_sample_data[accel_columns_y].values
                accel_z_data_real = real_sample_data[accel_columns_z].values

                # Plot data for real sample
                ax_real = axes[act_id, j * 2]  # Column for real dataset
                ax_real.plot(accel_x_data_real, color='r', alpha=0.7, label=f'{sensor} X (Real)')
                ax_real.plot(accel_y_data_real, color='g', alpha=0.7, label=f'{sensor} Y (Real)')
                ax_real.plot(accel_z_data_real, color='b', alpha=0.7, label=f'{sensor} Z (Real)')
                ax_real.set_title(f'Real: {activity_name} {j + 1}')
                ax_real.set_xlabel('Time')
                ax_real.legend()

            for j, synthetic_sample_index in enumerate(synthetic_sample_indices):
                synthetic_sample_data = synthetic_activity_data.loc[synthetic_sample_index]

                # Extract accelerometer data for synthetic sample
                accel_x_data_synthetic = synthetic_sample_data[accel_columns_x].values
                accel_y_data_synthetic = synthetic_sample_data[accel_columns_y].values
                accel_z_data_synthetic = synthetic_sample_data[accel_columns_z].values

                # Plot data for synthetic sample
                ax_synthetic = axes[act_id, j * 2 + 1]  # Column for synthetic dataset
                ax_synthetic.plot(accel_x_data_synthetic, color='r', alpha=0.7, label=f'{sensor} X (Synthetic)')
                ax_synthetic.plot(accel_y_data_synthetic, color='g', alpha=0.7, label=f'{sensor} Y (Synthetic)')
                ax_synthetic.plot(accel_z_data_synthetic, color='b', alpha=0.7, label=f'{sensor} Z (Synthetic)')
                ax_synthetic.set_title(f'Synthetic: {activity_name} {j + 1}')
                ax_synthetic.set_xlabel('Time')
                ax_synthetic.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def plot_random_samples_comparison_by_labels(self, num_samples=3):
        """
        Plots random samples of all sensor data for specific activities in both real and synthetic datasets.

        Parameters:
        num_samples (int): Number of random samples to be plotted for each activity.
        """
        # Ensure num_samples is an integer
        if not isinstance(num_samples, int):
            raise ValueError("num_samples must be an integer.")
        
        # Ensure num_samples is valid
        if num_samples <= 0:
            print("Number of samples must be greater than zero.")
            return

        # Create subplots for num_samples rows and 2 columns (one for each dataset)
        fig, axes = plt.subplots(nrows=len(self.activity_names), ncols=num_samples * 2, 
                                figsize=(5 * num_samples * 2, 5 * len(self.activity_names)), 
                                sharex=True, sharey=True)
        fig.suptitle('Data Samples Comparison: Real vs Synthetic', fontsize=16)

        for act_id, activity_name in enumerate(self.activity_names):
            # Filter data for the specific activity in both datasets
            real_activity_data = self.df_real[self.df_real[self.label_col] == act_id]
            synthetic_activity_data = self.df_synthetic[self.df_synthetic[self.label_col] == act_id]

            if real_activity_data.empty:
                print(f"No data available for activity in Real dataset: {activity_name}")
                continue
            
            if synthetic_activity_data.empty:
                print(f"No data available for activity in Synthetic dataset: {activity_name}")
                continue
            
            # Select random sample indices for both datasets
            real_sample_indices = np.random.choice(real_activity_data.index, min(num_samples, len(real_activity_data)), replace=False)
            synthetic_sample_indices = np.random.choice(synthetic_activity_data.index, min(num_samples, len(synthetic_activity_data)), replace=False)

            for j, real_sample_index in enumerate(real_sample_indices):
                real_sample_data = real_activity_data.loc[real_sample_index]

                # Plot all sensor data for real sample
                ax_real = axes[act_id, j * 2]  # Column for real dataset
                ax_real.plot(real_sample_data.values, color='c', alpha=0.7)
                ax_real.set_title(f'Real: {activity_name} {j + 1}')
                ax_real.set_xlabel('Time')
                ax_real.set_ylabel('Sensor Values')

            for j, synthetic_sample_index in enumerate(synthetic_sample_indices):
                synthetic_sample_data = synthetic_activity_data.loc[synthetic_sample_index]

                # Plot all sensor data for synthetic sample
                ax_synthetic = axes[act_id, j * 2 + 1]  # Column for synthetic dataset
                ax_synthetic.plot(synthetic_sample_data.values, color='m', alpha=0.7)
                ax_synthetic.set_title(f'Synthetic: {activity_name} {j + 1}')
                ax_synthetic.set_xlabel('Time')
                ax_synthetic.set_ylabel('Sensor Values')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig
    
    def plot_samplesT_by_label(self, num_samples=3):
        """
        Plots random samples of sensor data for each label, comparing real and synthetic datasets.

        Parameters:
        num_samples (int): Number of random samples to be plotted for each label.
        """
        # Ensure num_samples is an integer
        if not isinstance(num_samples, int):
            raise ValueError("num_samples must be an integer.")
        
        # Ensure num_samples is valid
        if num_samples <= 0:
            print("Number of samples must be greater than zero.")
            return
        
        # Create subplots: 2 columns (Real and Synthetic) for each label
        num_labels = len(self.activity_names)
        fig, axes = plt.subplots(nrows=num_labels, ncols=2, figsize=(10, num_labels * 5), sharex=True, sharey=True)
        fig.suptitle('Samples Comparison by Label: Real vs Synthetic', fontsize=16)

        for act_id, activity_name in enumerate(self.activity_names):
            # Filter data for the specific activity in both datasets
            real_activity_data = self.df_real[self.df_real[self.label_col] == act_id]
            synthetic_activity_data = self.df_synthetic[self.df_synthetic[self.label_col] == act_id]

            if real_activity_data.empty:
                print(f"No data available for activity in Real dataset: {activity_name}")
                continue
            
            if synthetic_activity_data.empty:
                print(f"No data available for activity in Synthetic dataset: {activity_name}")
                continue

            # Select random sample indices for both datasets
            real_sample_indices = np.random.choice(real_activity_data.index, min(num_samples, len(real_activity_data)), replace=False)
            synthetic_sample_indices = np.random.choice(synthetic_activity_data.index, min(num_samples, len(synthetic_activity_data)), replace=False)

            # Plot real samples
            for j, sample_index in enumerate(real_sample_indices):
                sample_data = real_activity_data.loc[sample_index]

                ax_real = axes[act_id, 0]  # First column for real dataset
                ax_real.plot(sample_data.values, alpha=0.7, label=f'Sample {j + 1}')
            
            ax_real.set_title(f'Real: {activity_name}')
            ax_real.set_xlabel('Time')
            ax_real.set_ylabel('Sensor Values')
            ax_real.legend()

            # Plot synthetic samples
            for j, sample_index in enumerate(synthetic_sample_indices):
                sample_data = synthetic_activity_data.loc[sample_index]

                ax_synthetic = axes[act_id, 1]  # Second column for synthetic dataset
                ax_synthetic.plot(sample_data.values, alpha=0.7, label=f'Sample {j + 1}')
            
            ax_synthetic.set_title(f'Synthetic: {activity_name}')
            ax_synthetic.set_xlabel('Time')
            ax_synthetic.set_ylabel('Sensor Values')
            ax_synthetic.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig
