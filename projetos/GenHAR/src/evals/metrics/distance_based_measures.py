from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, minkowski, cityblock, cosine, correlation
import pandas as pd
import numpy as np

def calculate_ed(ori_data, gen_data):
    if isinstance(ori_data, pd.DataFrame):
        ori_data = ori_data.to_numpy()
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.to_numpy()

    n_samples = min(ori_data.shape[0], gen_data.shape[0])
    n_series = ori_data.shape[2]
    
    distances = []
    for i in range(n_samples):
        sample_distance = np.mean([
            euclidean(ori_data[i, :, j], gen_data[i, :, j])
            for j in range(n_series)
        ])
        distances.append(sample_distance)
    
    return np.mean(distances)

def calculate_dtw(ori_data, gen_data):
    if isinstance(ori_data, pd.DataFrame):
        ori_data = ori_data.to_numpy()
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.to_numpy()

    n_samples = min(ori_data.shape[0], gen_data.shape[0])
    
    distances = []
    for i in range(n_samples):
        distance, _ = fastdtw(ori_data[i], gen_data[i], dist=euclidean)
        distances.append(distance)
        
    return np.mean(distances)

def calculate_minkowski(ori_data, gen_data, p=3):
    if isinstance(ori_data, pd.DataFrame):
        ori_data = ori_data.to_numpy()
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.to_numpy()

    n_samples = min(ori_data.shape[0], gen_data.shape[0])
    n_series = ori_data.shape[2]
    
    distances = []
    for i in range(n_samples):
        sample_distance = np.mean([
            minkowski(ori_data[i, :, j], gen_data[i, :, j], p=p)
            for j in range(n_series)
        ])
        distances.append(sample_distance)
    
    return np.mean(distances)

def calculate_manhattan(ori_data, gen_data):
    if isinstance(ori_data, pd.DataFrame):
        ori_data = ori_data.to_numpy()
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.to_numpy()

    n_samples = min(ori_data.shape[0], gen_data.shape[0])
    n_series = ori_data.shape[2]
    
    distances = []
    for i in range(n_samples):
        sample_distance = np.mean([
            cityblock(ori_data[i, :, j], gen_data[i, :, j])
            for j in range(n_series)
        ])
        distances.append(sample_distance)
    
    return np.mean(distances)

def calculate_cosine(ori_data, gen_data):
    if isinstance(ori_data, pd.DataFrame):
        ori_data = ori_data.to_numpy()
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.to_numpy()

    n_samples = min(ori_data.shape[0], gen_data.shape[0])
    n_series = ori_data.shape[2]
    
    distances = []
    for i in range(n_samples):
        sample_distance = np.mean([
            cosine(ori_data[i, :, j], gen_data[i, :, j])
            for j in range(n_series)
        ])
        distances.append(sample_distance)
    
    return np.mean(distances)

def calculate_pearson(ori_data, gen_data):
    if isinstance(ori_data, pd.DataFrame):
        ori_data = ori_data.to_numpy()
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.to_numpy()

    n_samples = min(ori_data.shape[0], gen_data.shape[0])
    n_series = ori_data.shape[2]
    
    distances = []
    for i in range(n_samples):
        sample_distance = np.mean([
            correlation(ori_data[i, :, j], gen_data[i, :, j])
            for j in range(n_series)
        ])
        distances.append(sample_distance)
    
    return np.mean(distances)
