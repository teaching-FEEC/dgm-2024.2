import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

def exact_match_score(original_data, synthetic_data): 
    exact_matches = synthetic_data.merge(original_data, how='inner')
    exact_match_score = len(exact_matches) / len(synthetic_data)    
    return exact_match_score

def neighbors_privacy_score(original_data, synthetic_data, k=1, threshold=0.1):
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(original_data)    
    distances, _ = nn_model.kneighbors(synthetic_data)    
    close_neighbors = distances <= threshold
    privacy_risk_count = close_neighbors.any(axis=1).sum()
    privacy_score = privacy_risk_count / len(synthetic_data)    
    return privacy_score

def membership_inference_score(original_data, synthetic_data):
    labeled_data = pd.concat([original_data.assign(label=1), synthetic_data.assign(label=0)])
    X = labeled_data.drop('label', axis=1)
    y = labeled_data['label']    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)    
    inference_score = clf.score(X_test, y_test)    
    return inference_score
