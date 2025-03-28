import numpy as np
class WeightInit:
    @staticmethod
    def zeros(size,seed=42):
        return np.zeros(size)  

    @staticmethod
    def random_uniform(size, low=-1, high=1, seed=42):
        np.random.seed(seed)
        return np.random.uniform(low, high, size) 

    @staticmethod
    def random_normal(size, mean=0, std=1, seed=42):
        np.random.seed(seed)
        return np.random.normal(mean, std, size)  

    @staticmethod
    def xavier_uniform(size, seed=42):
        np.random.seed(seed)
        limit = np.sqrt(6 / size[0])  
        return np.random.uniform(-limit, limit, size)  

    @staticmethod
    def xavier_normal(size, seed=42):
        np.random.seed(seed)
        std = np.sqrt(2 / size[0])
        return np.random.normal(0, std, size)  

    @staticmethod
    def he_uniform(size, seed=42):
        np.random.seed(seed)
        limit = np.sqrt(6 / size[0])
        return np.random.uniform(-limit, limit, size)  

    @staticmethod
    def he_normal(size, seed=42):
        np.random.seed(seed)
        std = np.sqrt(2 / size[0])
        return np.random.normal(0, std, size) 
