import numpy as np

def one_hot_encode(y):
    
    y = y.copy()
    y = y.astype(int).to_numpy()
    
    unique_classes = len(np.unique(y))
    
    encoded_y = np.zeros((y.shape[0], unique_classes))
    encoded_y[np.arange(y.shape[0]), y] = 1
    
    return encoded_y

def normalize(X):
    
    X = X.copy()
    X = X.to_numpy()
    
    normalized_X = X / 255.0
    
    return normalized_X