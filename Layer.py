from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import math_functions


class Layer:
    def forward(self, input: np.ndarray):
        raise NotImplemented

    def backward(self, output_gradient):
        raise NotImplemented

    def update(self, learning_rate):
        raise NotImplemented

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        raise NotImplemented

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        raise NotImplemented


@dataclass(slots=True)
class Softmax(Layer):
    input: np.ndarray | None = field(init=False, default=None)
    def forward(self, input: np.ndarray):
        self.input = input

        probs = np.exp(input - np.max(input, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def backward(self, expected):
        return self.input - expected

    def update(self, learning_rate):
        return

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        return

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        return

@dataclass(slots=True)
class Loss:
    function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(slots=True)
class Activation(Layer):
    function: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]

    input: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray):
        self.input = input
        return self.function(input)

    def backward(self, output_gradient):
        return self.derivative(self.input, output_gradient)

    def update(self, learning_rate):
        return

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        return

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        return


@dataclass(slots=True)
class FullyConnected(Layer):
    weights: np.ndarray = field()
    biases: np.ndarray = field()

    weights_gradient: np.ndarray = field(init=False)
    biases_gradient: np.ndarray = field(init=False)

    input: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        self._init_gradients()

    def _init_gradients(self):
        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

    def forward(self, input: np.ndarray):
        self.input = input
        return np.dot(input, self.weights.T) + self.biases.T

    def backward(self, output_gradient):
        self.weights_gradient += np.dot(output_gradient.T, self.input)
        self.biases_gradient += np.sum(output_gradient, axis=0)
        return np.dot(output_gradient, self.weights)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient

        self._init_gradients()

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        push(self.weights)
        push(self.biases)

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        self.biases = pop()
        self.weights = pop()
