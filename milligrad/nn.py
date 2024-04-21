import random

from .node import Node


class Neuron:
    def __init__(self, input_dim, activation="relu"):
        self.input_dim = input_dim
        self.weights = [Node(random.gauss()) for _ in range(input_dim)]
        self.bias = Node(random.random())
        self.activation = activation

    def __call__(self, x):
        assert (
            len(x) == self.input_dim
        ), f"dim(X) = {len(x)} not equal to input_dim = {self.input_dim}"
        value = sum(self.weights[i] * x[i] for i in range(self.input_dim)) + self.bias
        if self.activation == "relu":
            return value.relu()
        elif self.activation == "sigmoid":
            return value.sigmoid()
        else:
            assert False, f"Invalid activation {self.activation}"


class Layer:
    def __init__(self, input_dim, output_dim, activation="relu"):
        self.neurons = [
            Neuron(input_dim=input_dim, activation=activation)
            for _ in range(output_dim)
        ]

    def __call__(self, x):
        output = []
        for neuron in self.neurons:
            output.append(neuron(x))
        return output
