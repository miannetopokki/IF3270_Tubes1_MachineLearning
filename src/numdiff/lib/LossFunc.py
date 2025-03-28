import numpy as np
class LossFunction:
    def __init__(self, method):
        self.method = method
        self.compute = self.get_loss_function(method)
        self.derive = self.get_derivative_function(method)

    def get_loss_function(self, method):
        losses = {
            'mse': self.mse,
            'cce': self.cce,
            'bce': self.bce
        }
        if method not in losses:
            raise ValueError(f"Unknown loss function: {method}")
        return losses[method]

    def get_derivative_function(self, method):
        derivatives = {
            'mse': self.mse_derivative,
            'cce': self.cce_derivative,
            'bce': self.bce_derivative
        }
        if method not in derivatives:
            raise ValueError(f"Unknown loss function: {method}")
        return derivatives[method]

    def mse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def mse_derivative(self, y_true, y_pred):
        return y_pred - y_true  

    def cce(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

    def cce_derivative(self, y_true, y_pred):
        return y_pred - y_true 

    def bce(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

    def bce_derivative(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-8)  
