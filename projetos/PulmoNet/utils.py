import numpy as np


class MinMaxNormalize():
    '''
    Normaliza a imagem para o intervalo [0, 1]
    '''
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min())