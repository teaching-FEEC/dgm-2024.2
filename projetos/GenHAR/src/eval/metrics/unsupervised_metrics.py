import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mutual_info_score
from scipy.stats import entropy
from scipy.stats import zscore


class UnsupervisedLearningMetrics:
    def __init__(self, df):
        self.X = df.drop("label", axis=1)
        self.labels = df["label"]

    def silhouette(self):
        return silhouette_score(self.X, self.labels)

    def davies_bouldin(self):
        return davies_bouldin_score(self.X, self.labels)

    def calinski_harabasz(self):
        return calinski_harabasz_score(self.X, self.labels)

    def entropy_measure(self):
        class_counts = np.bincount(self.labels)
        probs = class_counts / np.sum(class_counts)
        return entropy(probs)

    def mutual_information(self):
        return mutual_info_score(self.labels, self.labels)

    def a_distance(self):
        # Implementação da A-distance
        pass

    def h_delta_h_divergence(self):
        # Implementação da H∆H-divergence
        pass

    def mcd(self):
        # Implementação da Minimum Classification Difference
        pass

    def mdd(self):
        # Implementação da Minimum Distance Difference
        pass

    def dev(self):
        # Implementação da Divergence Estimation via Variance
        pass

    def snd(self):
        # Implementação da Similarity Network Dissimilarity
        pass

    def ism(self):
        # Implementação do Improvements on Similarity Measures
        pass

    def acm(self):
        # Implementação do Adapted Classification Measures
        pass

    def evaluate(self):
        results = {
            'Silhouette Score': self.silhouette(),
            'Davies-Bouldin Index': self.davies_bouldin(),
            'Calinski-Harabasz Index': self.calinski_harabasz(),
            'Entropy': self.entropy_measure(),
            'Mutual Information': self.mutual_information(),
            'A-distance': self.a_distance(),
            'H∆H-divergence': self.h_delta_h_divergence(),
            'MCD': self.mcd(),
            'MDD': self.mdd(),
            'DEV': self.dev(),
            'SND': self.snd(),
            'ISM': self.ism(),
            'ACM': self.acm(),
        }
        return results
    
    
    



    def zscore_outliers(self, threshold=3):
        z_scores = np.abs(zscore(self.X))
        outliers = (z_scores > threshold).any(axis=1)
        return self.X[outliers], self.labels[outliers]


# Exemplo de uso:
# df = ... # Seu DataFrame com a coluna 'label'
# metrics = UnsupervisedLearningMetrics(df)
# results = metrics.evaluate()
# print(results)
