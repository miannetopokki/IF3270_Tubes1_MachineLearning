from lib.Weight import *
from lib.Activation import *
from lib.LossFunc import *
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
from tqdm import tqdm
import time


class Layer:
    def __init__(self, input_size, n_neurons, activation, 
                 weight_init='random_normal', bias_init='zeros', 
                 seed=42, **kwargs):

        self.input_size = input_size
        self.n_neurons = n_neurons
        self.activation = Activation(activation)  
        self.seed = seed
        self.weights = self.initialize_weights(weight_init, seed, **kwargs)
        self.biases = self.initialize_biases(bias_init, seed, **kwargs)



    def initialize_weights(self, method, seed, **kwargs):
        if hasattr(WeightInit, method):
            return getattr(WeightInit, method)(size=(self.input_size, self.n_neurons), seed=seed, **kwargs)  
        else:
            raise ValueError(f"init bobot tidak diketahui: {method}")

    def initialize_biases(self, method, seed, **kwargs):
        if hasattr(WeightInit, method):
            return getattr(WeightInit, method)(size=(1, self.n_neurons), seed=seed, **kwargs)  
        else:
            raise ValueError(f"init bias tidak diketahui: {method}")


class MLP:
    def __init__(self, layers,loss_function, lr=0.01, regularization= None,reg_lambda = 0.001):
        
        self.lr = lr
        self.num_layers = len(layers)
        self.layers = layers
        self.loss_function = LossFunction(loss_function)
        self.loss_graph = []
        self.valid_graph = []
        self.regularization = regularization  
        self.reg_lambda = reg_lambda          
        self.weights_history = {i: [] for i in range(self.num_layers)}
        self.gradients_history = {i: [] for i in range(self.num_layers)}
        self.train_history = []

        
    def forward(self, X):
        self.activations = [X]
        for i in range(self.num_layers):
            z = np.dot(self.activations[-1], self.layers[i].weights) + self.layers[i].biases
            a = self.layers[i].activation.activate(z)
            self.activations.append(a)

        return a
    def backward(self, X, y_true):
        m = X.shape[0]
        y_pred = self.activations[-1]

        if self.loss_function.method == 'mse':
            # perkalian turunan rantai
            delta = self.loss_function.derive(y_true, y_pred) * self.layers[-1].activation.derive(y_pred)
        elif self.loss_function.method in ['cce', 'bce']: 
            # khasus khusus untuk cce + softmax, atau bce + sigmoid, pakai turunan langsung
            delta = self.loss_function.derive(y_true, y_pred)

        errors = [delta]  

        # backprop
        for i in range(self.num_layers - 1, 0, -1):
            # print(f"Layer {i} - Activation: {self.layers[i-1].activation.method}")
            # dA = error dari layer di atas, dikali turunan aktivasi dari layer ini
            dA = np.dot(errors[-1], self.layers[i].weights.T)  
            delta = dA * self.layers[i-1].activation.derive(self.activations[i])  
            errors.append(delta)

        errors.reverse()  # Urutkan dari input → output

        # grad descent
        for i in range(self.num_layers):
            dW = (1 / m) * np.dot(self.activations[i].T, errors[i])
            db = (1 / m) * np.sum(errors[i], axis=0, keepdims=True)

            # regularisasi
            if self.regularization == 'l2':
                dW += self.reg_lambda * self.layers[i].weights
            elif self.regularization == 'l1':
                dW += self.reg_lambda * np.sign(self.layers[i].weights)


            self.layers[i].weights -= self.lr * dW
            self.layers[i].biases -= self.lr * db

            self.weights_history[i].append(self.layers[i].weights.flatten())
            self.gradients_history[i].append(dW.flatten())
        
    # print("backward pass done")
    def train(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=64,verbose= 1):
        
        self.train_history = []
        start_time = time.time()
        
        m = X.shape[0]
        
        progress_bar = tqdm(total=epochs, desc="Training", unit="epoch", position=0, leave=True) if verbose else None

        for epoch in range(epochs):
            perm = np.random.permutation(m)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch)

            y_pred_train = self.forward(X)
            loss_train = self.loss_function.compute(y,y_pred_train)
            acc_train = self.accuracy(X, y)
            self.loss_graph.append(loss_train)

            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                loss_val = self.loss_function.compute(y_val,y_pred_val)
                vall_acc = self.accuracy(X_val, y_val)
                self.valid_graph.append(loss_val)
                if verbose:
                    progress_bar.set_postfix(train_loss=f"{loss_train:.4f}", 
                                            val_loss=f"{loss_val:.4f}", 
                                            train_accuracy=f"{acc_train:.2f}%",
                                            val_accuracy=f"{vall_acc:.2f}%")
            elif verbose:
                progress_bar.set_postfix(train_loss=f"{loss_train:.4f}", 
                                        train_accuracy=f"{acc_train:.2f}%")
                
            
            if verbose:
                progress_bar.update(1)
                
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': loss_train,
                'val_loss': loss_val if X_val is not None and y_val is not None else None,
                'train_accuracy': acc_train,
                'val_accuracy': vall_acc if X_val is not None and y_val is not None else None,
                'time_taken': time.time() - start_time
            })

        if verbose:
            progress_bar.close()
                
    def accuracy(self, X, y_true):
        y_pred = self.forward(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(y_pred_classes == y_true_classes) * 100

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def plot_loss(self):
        plt.plot(self.loss_graph, label="Training Loss",color = "red")
        plt.plot(self.valid_graph,label = "Validation Loss",color = "blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
    def print_train_history(self):
        
        if len(self.train_history) == 0:
            print("No training history available.")
            return
        
        has_validation = self.train_history[0]['val_loss'] is not None

        if has_validation:
            print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Train Acc':<12} | {'Val Acc':<10} | {'Time':<10}")
            print("-" * 75)
        else:
            print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Time':<10}")
            print("-" * 50)

        for i, row in enumerate(self.train_history):

            epoch = row['epoch']
            train_loss = f"{row['train_loss']:.4f}"
            val_loss = f"{row['val_loss']:.4f}" if row['val_loss'] is not None else "-"
            train_acc = f"{row['train_accuracy']:.2f}%" if row['train_accuracy'] is not None else "-"
            val_acc = f"{row['val_accuracy']:.2f}%" if row['val_accuracy'] is not None else "-"
            time_taken = f"{row['time_taken']:.2f}s"

            if has_validation:
                print(f"{epoch:<8} | {train_loss:<12} | {val_loss:<10} | {train_acc:<12} | {val_acc:<10} | {time_taken:<10}")
            else:
                print(f"{epoch:<8} | {train_loss:<12} | {train_acc:<12} | {time_taken:<10}")

    

    def plot_weight_distribution(self):
        fig, axes = plt.subplots(1, self.num_layers, figsize=(15, 5))
        for i in range(self.num_layers):
            axes[i].hist(self.weights_history[i][-1], bins=30, alpha=0.7, color='b')
            axes[i].set_title(f'Layer {i+1} Weights')
        plt.show()

    def plot_gradient_distribution(self):
        fig, axes = plt.subplots(1, self.num_layers, figsize=(15, 5))
        for i in range(self.num_layers):
            axes[i].hist(self.gradients_history[i][-1], bins=30, alpha=0.7, color='r')
            axes[i].set_title(f'Layer {i+1} Gradients')
        plt.show()
    def save(self, filepath):
        model_state = {
            'num_layers': self.num_layers,
            'lr': self.lr,
            'loss_function_method': self.loss_function.method,
            'loss_graph': self.loss_graph,
            'valid_graph': self.valid_graph,
            'layers_data': []
        }
        
        for i, layer in enumerate(self.layers):
            layer_data = {
                'input_size': layer.input_size,
                'n_neurons': layer.n_neurons,
                'activation_method': layer.activation.method,
                'weights': layer.weights,
                'biases': layer.biases,
                'seed': layer.seed
            }
            model_state['layers_data'].append(layer_data)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        print(f"Model successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath):

        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        layers = []
        for layer_data in model_state['layers_data']:
            layer = Layer(
                input_size=layer_data['input_size'],
                n_neurons=layer_data['n_neurons'],
                activation=layer_data['activation_method'],
                seed=layer_data['seed']
            )
            layer.weights = layer_data['weights']
            layer.biases = layer_data['biases']
            layers.append(layer)
        
        model = cls(layers, model_state['loss_function_method'], lr=model_state['lr'])
        
        model.loss_graph = model_state['loss_graph']
        model.valid_graph = model_state['valid_graph']
        
        print(f"Model successfully loaded from {filepath}")
        return model
