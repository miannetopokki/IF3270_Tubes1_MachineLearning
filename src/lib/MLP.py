

import random
from lib.value import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, activation="tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
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
    def __init__(self, nin, nout, activation="tanh"):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts, activations=None):
        if activations is None:
            activations = ["tanh"] * len(nouts)

        if len(activations) != len(nouts):
            raise ValueError("activations list must match the number of layers (nouts).")

        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], activation=activations[i]) for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
