import math
from typing import List

from graphviz import Digraph


class Node:
    def __init__(self, value, op=None, parents: List["Node"] = []):
        self.value = value
        self.op = op
        self.grad = 0
        self.parents = parents

    def __add__(self, other):
        return Add.apply(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Sub.apply(self, other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return Neg.apply(Neg, self)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        return Pow.apply(self, other)

    def __rpow__(self, other):
        return Pow.apply(other, self)

    def sigmoid(self):
        return Sigmoid.apply(Sigmoid, self)

    def relu(self):
        return ReLU.apply(ReLU, self)

    def log(self):
        return Log.apply(Log, self)

    def topological_sort(self):
        list = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build(parent)
                list.append(node)

        build(self)
        return list

    def backward(self, seed=1):
        self.grad = seed

        for node in reversed(self.topological_sort()):
            if node.op:
                node.op.backward()

    def zero_grad(self):
        for node in reversed(self.topological_sort()):
            node.grad = 0

    def step(self, learning_rate):
        for nodes in reversed(self.topological_sort()):
            nodes.value -= self.grad * learning_rate

    def __repr__(self):
        return f"Node(value={self.value}, op={self.op.__class__.__name__ if self.op else None}, grad={self.grad}, parents={self.parents})"

    def visualize(self, graph=None, visited=None):
        if graph is None:
            graph = Digraph(format="png")
        if visited is None:
            visited = set()

        if self in visited:
            return graph
        visited.add(self)

        if self.op:
            label = (
                f"{self.op.__class__.__name__}\nvalue: {self.value}\ngrad: {self.grad}"
            )
        else:
            label = f"value: {self.value}\ngrad: {self.grad}"

        graph.node(name=str(id(self)), label=label)

        for parent in self.parents:
            parent.visualize(graph, visited)
            graph.edge(str(id(parent)), str(id(self)))

        return graph


def track(a):
    return a if type(a) == Node else Node(a)


class UnaryOp:
    def __init__(self, a: Node):
        self.a = a

    def apply(cls, a):
        return cls(a).forward()


class BinaryOp:
    def __init__(self, a, b):
        self.a = track(a)
        self.b = track(b)

    @classmethod
    def apply(cls, a, b):
        return cls(a, b).forward()


class Add(BinaryOp):
    def forward(self):
        self.c = Node(self.a.value + self.b.value, self, parents=[self.a, self.b])
        return self.c

    def backward(self):
        self.a.grad += 1 * self.c.grad
        self.b.grad += 1 * self.c.grad


class Sub(BinaryOp):
    def forward(self):
        self.c = Node(self.a.value - self.b.value, self, parents=[self.a, self.b])
        return self.c

    def backward(self):
        self.a.grad += 1 * self.c.grad
        self.b.grad += -1 * self.c.grad


class Mul(BinaryOp):
    def forward(self):
        self.c = Node(self.a.value * self.b.value, self, parents=[self.a, self.b])
        return self.c

    def backward(self):
        self.a.grad += self.b.value * self.c.grad
        self.b.grad += self.a.value * self.c.grad


class Pow(BinaryOp):
    def forward(self):
        self.c = Node(self.a.value**self.b.value, self, parents=[self.a, self.b])
        return self.c

    def backward(self):
        self.a.grad += self.b.value * self.a.value ** (self.b.value - 1) * self.c.grad
        # self.b.grad += self.c.value * math.log(self.a.value) * self.c.grad


class Neg(UnaryOp):
    def forward(self):
        self.b = Node(-self.a.value, self, parents=[self.a])
        return self.b

    def backward(self):
        self.a.grad += -1 * self.b.grad


class ReLU(UnaryOp):
    def forward(self):
        self.b = Node(max(0, self.a.value), self, parents=[self.a])
        return self.b

    def backward(self):
        self.a.grad += (1 if self.a.value > 0 else 0) * self.b.grad


class Sigmoid(UnaryOp):
    def forward(self):
        self.b = Node(1 / (1 + math.exp(-self.a.value)), self, parents=[self.a])
        return self.b

    def backward(self):
        self.a.grad += self.b.value * (1 - self.b.value) * self.b.grad


class Log(UnaryOp):
    def forward(self):
        try:
            self.b = Node(math.log(self.a.value + 1e-10), self, parents=[self.a])
        except:
            print(self.a.value)
            quit()
        return self.b

    def backward(self):
        self.a.grad += (1 / self.b.value) * self.b.grad
