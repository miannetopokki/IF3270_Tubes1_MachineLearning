import math
import random
from lib.value import Value


class Weight:
    def __init__(self, method, seed, nin, upper=None, lower=None, mean=None, variance=None):
        self.method = method
        self.nin = nin
        self.upper = upper
        self.lower = lower
        self.mean = mean
        self.variance = variance
        self.rng = random.Random(seed) 

    def __call__(self):
        if self.method == "uniform":
            if self.lower is None or self.upper is None:
                raise ValueError("Lower and upper bounds must be provided for uniform distribution.")
            return [self.rng.uniform(self.lower, self.upper) for _ in range(self.nin)]
        
        elif self.method == "normal":
            if self.mean is None or self.variance is None:
                raise ValueError("Mean and variance must be provided for normal distribution.")
            stddev = math.sqrt(self.variance)
            return [self.rng.gauss(self.mean, stddev) for _ in range(self.nin)]
        
        elif self.method == "zero":
            return [0 for _ in range(self.nin)]
        
        else:
            raise ValueError("Invalid weight initialization method.")


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, activation="tanh",weight: Weight=None):
        raw_weights = weight() if weight is not None else [random.uniform(-1, 1) for _ in range(nin)]
        self.w = [Value(w) for w in raw_weights]
        #ini bobot bias
        self.b = Value(0)
        self.activation = activation.lower()

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.activation == "relu":
            return act.relu()
        elif self.activation == "linear":
            return act
        elif self.activation == "tanh":
            return act.tanh()
        else:
            raise ValueError()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation.capitalize()}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, activation="tanh", weight: Weight=None):
        self.neurons = [Neuron(nin, activation=activation, weight=weight) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts, activations=None, weight: Weight=None):
        #jika activation tidak diisi maka akan diisi dengan nilai default yaitu tanh
        if activations is None:
            activations = ["tanh"] * len(nouts)

        if len(activations) != len(nouts):
            raise ValueError("activations list must match the number of layers (nouts).")

        sz = [nin] + nouts
        print("sz: " ,sz)
        self.layers = [
            Layer(sz[i], sz[i + 1], activation=activations[i],weight=weight) for i in range(len(nouts))
        ]

        self.weight = weight
        self.inputlayer = nin

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"inputx : {self.inputlayer} MLP of [{', '.join(str(layer) for layer in self.layers)}]"
