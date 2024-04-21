import math
import os
import random

import graphviz
from datasets import load_dataset
from tqdm import tqdm


class Node:
    def __init__(self, value, op="", parents=[]):
        self.value = value
        self.op = op
        self.parents = parents
        self.grad = 0

    def __add__(self, other):
        if isinstance(other, Node):
            return Node(value=self.value + other.value, op="add", parents=[self, other])
        o = Node(other)
        return Node(value=self.value + o.value, op="add", parents=[self, o])

    def __sub__(self, other):
        if isinstance(other, Node):
            return Node(value=self.value - other.value, op="sub", parents=[self, other])
        o = Node(other)
        return Node(value=self.value - o.value, op="sub", parents=[self, o])

    def __mul__(self, other):
        if isinstance(other, Node):
            return Node(value=self.value * other.value, op="mul", parents=[self, other])
        o = Node(other)
        return Node(value=self.value * o.value, op="mul", parents=[self, o])

    def __truediv__(self, other):
        if isinstance(other, Node):
            return Node(value=self.value / other.value, op="div", parents=[self, other])
        o = Node(other)
        return Node(value=self.value / o.value, op="div", parents=[self, o])

    def relu(self):
        return Node(value=max(0, self.value), op="relu", parents=[self])

    def sigmoid(self):
        return Node(value=1 / (1 + math.exp(-self.value)), op="sigmoid", parents=[self])
    
    def softmax(self):
        return Node(value=math.exp(self.value) / sum(math.exp(x.value) for x in self.parents), op="softmax", parents=[self])

    def __neg__(self):
        return Node(value=-self.value, op="neg", parents=[self])

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def backward(self, grad=1, visited=set()):
        self.grad += grad
        if self not in visited:
            visited.add(self)
            if self.op == "add":
                # d/da a+b = 1
                self.parents[0].backward(1 * self.grad, visited)

                # d/db a+b = 1
                self.parents[1].backward(1 * self.grad, visited)
            elif self.op == "mul":
                # d/da a*b = b
                self.parents[0].backward(self.parents[1].value * self.grad, visited)

                # d/db a*b = a
                self.parents[1].backward(self.parents[0].value * self.grad, visited)
            elif self.op == "sub":
                # d/da a-b = 1
                self.parents[0].backward(1 * self.grad, visited)

                # d/db a-b = -1
                self.parents[1].backward(-1 * self.grad, visited)
            elif self.op == "div":
                # d/da a/b = 1/b
                self.parents[0].backward((1 / self.parents[1].value) * self.grad, visited)

                # d/db a/b = -a/b^2
                self.parents[1].backward(
                    (-self.parents[0].value / self.parents[1].value ** 2) * self.grad, visited
                )
            elif self.op == "relu":
                # d/da relu(a) = 0 if a < 0 else 1
                self.parents[0].backward((1 if self.value > 0 else 0) * self.grad, visited)
            elif self.op == "sigmoid":
                # d/da sigmoid(a) = d/da 1/(1+e^(-a)) = sigmoid(a) * (1 - sigmoid(a))
                self.parents[0].backward(self.value * (1 - self.value) * self.grad, visited)
            elif self.op == "softmax":
                # d/da softmax(a) = e^a / sum(e^a)
                self.parents[0].backward(self.value * (1 - self.value) * self.grad, visited)
            elif self.op == "neg":
                # d/da -a = -1

                self.parents[0].backward(-1 * self.grad, visited)

    def zero_grad(self, visited=set()):
        self.grad = 0
        if self not in visited:
            visited.add(self)
            for parent in self.parents:
                parent.zero_grad(visited)

    def step(self, learning_rate, visited=set()):
        self.value -= learning_rate * self.grad
        if self not in visited:
            visited.add(self)
            for parent in self.parents:
                parent.step(learning_rate, visited)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Value(value={self.value}, op={self.op}, grad={self.grad})"


def graph(node):
    dot = graphviz.Digraph(comment="Forward")

    def helper(child):
        dot.node(str(id(child)), str(child))
        for parent in child.parents:
            helper(parent)
            dot.edge(str(id(parent)), str(id(child)))

    helper(node)

    dot.render(format="png")


class Neuron:
    def __init__(self, input_dim, activation="relu"):
        self.input_dim = input_dim
        self.weights = [Node(random.random()) for _ in range(input_dim)]
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
        elif self.activation == "softmax":
            return value.softmax()
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


class Model:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.l1 = Layer(input_dim, hidden_dim, activation="relu")
        self.l2 = Layer(hidden_dim, output_dim, activation="sigmoid")

    def __call__(self, x):
        return self.l2(self.l1(x))


def mse(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    s = 0
    for y_t, y_p in zip(y_true, y_pred):
        d = y_t - y_p
        s += d * d

    return s / len(y_pred)


def image_to_list(im):
    pixels = list(im.getdata())
    for i in range(len(pixels)):
        pixels[i] /= 255
    return pixels


def make_one_hot(index):
    vector = [0 for _ in range(10)]
    vector[index] = 1
    return vector


def main():
    dataset = load_dataset("mnist").shuffle()
    model = Model(28 * 28, 64, 10)

    for example in dataset["train"]:
        image = image_to_list(example["image"])
        label = make_one_hot(example["label"])
        print(label)
        output = model(image)
        loss = mse(y_true=label, y_pred=output)
        loss.backward()
        loss.step(learning_rate=1e-3)
        loss.zero_grad()


if __name__ == "__main__":
    main()
