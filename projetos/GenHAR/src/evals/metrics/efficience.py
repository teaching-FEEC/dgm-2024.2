import time,psutil,csv,json
from pathlib import Path
import GPUtil

class EfficiencyMonitor:
    def __init__(self):
        self.start_time = None
        self.start_mem = None
        self.start_gpu_mem = None
        self.cpu_usage = []
        self.gpu_usage = []

    def start(self):
        # Marca o tempo inicial e a memória inicial
        self.start_time = time.time()
        self.start_mem = psutil.Process().memory_info().rss / (1024 ** 2)  # Convertendo para MB
        self.cpu_usage = [psutil.cpu_percent(interval=None)]
        if GPUtil:
            self.start_gpu_mem = sum(gpu.memoryUsed for gpu in GPUtil.getGPUs()) if GPUtil else 0
            self.gpu_usage = [gpu.load * 100 for gpu in GPUtil.getGPUs()]

    def stop(self):
        # Calcula o tempo total, a memória utilizada e o uso de CPU/GPU
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 ** 2)  # Convertendo para MB
        total_time = end_time - self.start_time
        mem_used = end_mem - self.start_mem

        avg_cpu_usage = sum(self.cpu_usage) / len(self.cpu_usage)
        results = {
            "Tempo de execução (s)": total_time,
            "Memória utilizada (MB)": mem_used,
            "Uso médio de CPU (%)": avg_cpu_usage,
        }

        # Se disponível, calcula o uso de GPU
        if GPUtil:
            end_gpu_mem = sum(gpu.memoryUsed for gpu in GPUtil.getGPUs())
            gpu_mem_used = end_gpu_mem - (self.start_gpu_mem or 0)
            avg_gpu_usage = sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0
            results.update({
                "Memória GPU utilizada (MB)": gpu_mem_used,
                "Uso médio de GPU (%)": avg_gpu_usage,
            })

        return results

    def save_metrics(self, metrics, file_path, format="csv"):
        """
        Salva as métricas em um arquivo CSV ou JSON.

        :param metrics: Dicionário com as métricas a serem salvas.
        :param file_path: Caminho do arquivo para salvar as métricas.
        :param format: Formato do arquivo ("csv" ou "json").
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        if format == "csv":
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(metrics.keys())
                writer.writerow(metrics.values())
        elif format == "json":
            with open(file_path, mode='a') as file:
                json.dump(metrics, file)
                file.write("\n")  
