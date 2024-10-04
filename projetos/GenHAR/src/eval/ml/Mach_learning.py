import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfReader, PdfWriter
from typing import Dict
from io import BytesIO

class ModelEvaluator:
    def __init__(self, classifiers: Dict[str, any], 
                 df_train: pd.DataFrame, df_test: pd.DataFrame, 
                 df_val: pd.DataFrame, df_synthetic: pd.DataFrame,
                 label: str, generator_name: str, dataset_name: str, 
                 transformation_name: str):
        self.classifiers = classifiers
        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val
        self.df_synthetic = df_synthetic
        self.label = label  # Column name for the labels
        self.generator_name = generator_name
        self.dataset_name = dataset_name
        self.transformation_name = transformation_name

        # Separar features e labels
        self.X_train_real = df_train.drop(columns=[label]).values
        self.y_train_real = df_train[label].values
        self.X_test = df_test.drop(columns=[label]).values
        self.y_test = df_test[label].values
        self.X_synthetic = df_synthetic.drop(columns=[label]).values
        self.y_synthetic = df_synthetic[label].values

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
        figs=[]
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
            #plt.savefig(os.path.join(self.model_dir, f'{metric}_comparison.png'))
            figs.append(fig)
            #plt.close()
        return figs

    def plot_boxplots(self, metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        metrics_labels = ['accuracy', 'precision', 'recall', 'f1-score']
        figs=[]
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
            #plt.savefig(os.path.join(self.model_dir, f'{metric}_boxplot.png'))
            figs.append(fig)
            plt.close()
        return figs
            


    def save_metrics_to_csv(self, metrics: Dict[str, Dict[str, Dict[str, float]]], folder_name: str) -> None:
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
        csv_path = os.path.join(folder_name,self.model_dir, "/evaluation_metrics.csv")
        df.to_csv(csv_path,mode='w' ,index=False)
        print(f"Metrics saved to {csv_path}")

    def save_metrics_to_pdf(self, metrics: Dict[str, Dict[str, Dict[str, float]]], pdf_path: str) -> None:
        # Converte o dicionário de métricas em um DataFrame
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
        
        df = pd.DataFrame(data)  # Converte o dicionário em DataFrame

        # Verifica se o PDF já existe

        # Se o arquivo já existir, adicione uma nova página com métricas, caso contrário, crie um novo arquivo
        if os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            writer = PdfWriter()

            # Copia as páginas do PDF existente
            for page_num in range(len(reader.pages)):
                writer.add_page(reader.pages[page_num])

        else:
            writer = PdfWriter()

        # Cria uma nova página com as métricas
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from io import BytesIO

        packet = BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)

        # Adiciona o título e as métricas
        text_object = can.beginText(40, 750)
        text_object.setFont("Helvetica", 12)
        text_object.textLine(f"Evaluation Metrics for {self.dataset_name}")
        text_object.moveCursor(0, 20)
        
        for index, row in df.iterrows():
            row_text = ', '.join([f"{col}: {row[col]}" for col in df.columns])
            text_object.textLine(row_text)

        can.drawText(text_object)
        can.save()

        packet.seek(0)

        # Lê a nova página
        new_pdf_reader = PdfReader(packet)
        writer.add_page(new_pdf_reader.pages[0])

        # Salva o PDF atualizado
        with open(pdf_path, "wb") as output_pdf:
            writer.write(output_pdf)

        print(f"Métricas salvas ou adicionadas ao arquivo PDF: {pdf_path}")



