import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from typing import Dict

class ModelEvaluator:
    def __init__(self, classifiers: Dict[str, any], 
                 X_train_real: np.ndarray, y_train_real: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, 
                 X_synthetic: np.ndarray, y_synthetic: np.ndarray,
                 generator_name: str, dataset_name: str, transformation_name: str):
        self.classifiers = classifiers
        self.X_train_real = X_train_real
        self.y_train_real = y_train_real
        self.X_test = X_test
        self.y_test = y_test
        self.X_synthetic = X_synthetic
        self.y_synthetic = y_synthetic
        self.generator_name = generator_name
        self.dataset_name = dataset_name
        self.transformation_name = transformation_name

        # Dados misturados
        self.X_train_mixed = np.vstack((self.X_train_real, self.X_synthetic))
        self.y_train_mixed = np.hstack((self.y_train_real, self.y_synthetic))

        # Criar diretórios para salvar os relatórios
        self.base_dir = 'reports'
        self.dataset_dir = os.path.join(self.base_dir, self.dataset_name)
        self.model_dir = os.path.join(self.dataset_dir, self.generator_name)
        os.makedirs(self.model_dir, exist_ok=True)

    def evaluate_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray, classifier: any) -> Dict[str, float]:
        clf = classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1-score': report['1']['f1-score']
        }

    def evaluate_all_models(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        metrics = {}
        for name, clf in self.classifiers.items():
            metrics[name] = {
                'real': self.evaluate_model(self.X_train_real, self.y_train_real, self.X_test, self.y_test, clf),
                'synthetic': self.evaluate_model(self.X_synthetic, self.y_synthetic, self.X_test, self.y_test, clf),
                'mixed': self.evaluate_model(self.X_train_mixed, self.y_train_mixed, self.X_test, self.y_test, clf)
            }
        return metrics

    def plot_metrics(self, metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        metrics_labels = ['accuracy', 'precision', 'recall', 'f1-score']
        for metric in metrics_labels:
            fig, ax = plt.subplots(figsize=(14, 8))
            x = np.arange(len(self.classifiers))  # Localizações dos rótulos
            width = 0.2  # Largura das barras

            for i, (name, metrics_dict) in enumerate(metrics.items()):
                real_values = metrics_dict['real'][metric]
                synthetic_values = metrics_dict['synthetic'][metric]
                mixed_values = metrics_dict['mixed'][metric]

                ax.bar(x[i] - width, real_values, width, label=f'{name} (Reais)')
                ax.bar(x[i], synthetic_values, width, label=f'{name} (Sintéticos)')
                ax.bar(x[i] + width, mixed_values, width, label=f'{name} (Misturados)')

            # Adicionar rótulos e título
            ax.set_xlabel('Classificadores')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{self.dataset_name} - {self.generator_name} - {self.transformation_name} - Comparação dos Classificadores para {metric.capitalize()}')
            ax.set_xticks(x)
            ax.set_xticklabels(self.classifiers.keys(), rotation=45, ha='right')
            ax.legend()

            # Ajustar layout e salvar o gráfico
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f'{metric}_comparison.png'))
            plt.close()

    def plot_boxplots(self, metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        metrics_labels = ['accuracy', 'precision', 'recall', 'f1-score']
        for metric in metrics_labels:
            fig, ax = plt.subplots(figsize=(14, 8))

            # Preparar os dados para o boxplot
            data_to_plot = {
                'Real': [],
                'Synthetic': [],
                'Mixed': []
            }

            for name, metrics_dict in metrics.items():
                data_to_plot['Real'].append(metrics_dict['real'][metric])
                data_to_plot['Synthetic'].append(metrics_dict['synthetic'][metric])
                data_to_plot['Mixed'].append(metrics_dict['mixed'][metric])

            # Criar boxplot
            ax.boxplot([data_to_plot['Real'], data_to_plot['Synthetic'], data_to_plot['Mixed']], 
                       labels=['Real', 'Synthetic', 'Mixed'])
            ax.set_xlabel('Tipo de Dados')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{self.dataset_name} - {self.generator_name} - {self.transformation_name} - Boxplot de {metric.capitalize()} por Tipo de Dados')

            # Ajustar layout e salvar o gráfico
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f'{metric}_boxplot.png'))
            plt.close()

    def save_metrics_to_csv(self, metrics: Dict[str, Dict[str, Dict[str, float]]], filename: str) -> None:
        data = []
        for name, metrics_dict in metrics.items():
            for data_type in ['real', 'synthetic', 'mixed']:
                data.append({
                    'Classifier': name,
                    'Data Type': data_type.capitalize(),
                    'Generator': self.generator_name,
                    'Dataset': self.dataset_name,
                    'Transformation': self.transformation_name,
                    'Accuracy': metrics_dict[data_type]['accuracy'],
                    'Precision': metrics_dict[data_type]['precision'],
                    'Recall': metrics_dict[data_type]['recall'],
                    'F1 Score': metrics_dict[data_type]['f1-score']
                })
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.model_dir, filename)
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")