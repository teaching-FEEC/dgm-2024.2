from scipy import stats
import numpy as np

#Teste de Kolmogorov-Smirnov (KS)
def ks_test(X_train, X_gen):
    # Verificar se as duas matrizes têm o mesmo número de dimensões
    if X_train.shape[1] != X_gen.shape[1]:
        raise ValueError("X_train e X_gen devem ter o mesmo número de características.")

    # Loop através de cada dimensão/característica
    results = {}
    for i in range(X_train.shape[1]):
        # Extraindo a dimensão i de X_train e X_gen
        feature_train = X_train[:, i]
        feature_gen = X_gen[:, i]

        # Aplicar o Teste KS de duas amostras para a dimensão i
        statistic, p_value = stats.ks_2samp(feature_train, feature_gen)

        # Armazenar o resultado para a dimensão i
        results[f'Feature {i}'] = {'KS Statistic': statistic, 'p-value': p_value}

    return results


#from tsfresh import extract_features, select_features




