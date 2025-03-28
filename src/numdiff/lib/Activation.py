import numpy as np
class Activation:
    def __init__(self, method):
        self.method = method
        self.activate = self.get_activation_function(method)
        self.derive = self.get_derivative_function(method)

    def get_activation_function(self, method):
        activations = {
            'linear': self.linear,
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'softmax': self.softmax
        }
        if method not in activations:
            raise ValueError(f"Unknown activation method: {method}")
        return activations[method]

    def get_derivative_function(self, method):
        derivatives = {
            'linear': self.linear_derivative,
            'relu': self.relu_derivative,
            'sigmoid': self.sigmoid_derivative,
            'tanh': self.tanh_derivative,
            'softmax': self.softmax_derivative
        }
        if method not in derivatives:
            raise ValueError(f"Unknown activation method: {method}")
        return derivatives[method]

    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return np.ones_like(x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # print("envoked relu_derivative")
        return (x > 0).astype(float)
    

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def softmax(self, x):
        # print("envoked softmax derivative")
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def softmax_derivative(self, x):
        return x * (1 - x)