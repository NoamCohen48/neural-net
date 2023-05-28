from dataclasses import dataclass, field
from typing import Callable

import numpy as np


class Layer:
    def forward(self, input: np.ndarray):
        raise NotImplemented

    def backward(self, output_gradient, learning_rate):
        raise NotImplemented

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        raise NotImplemented

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        raise NotImplemented


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

    def backward(self, output_gradient, learning_rate):
        return self.derivative(self.input, output_gradient)

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        return

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        return


@dataclass(slots=True)
class FullyConnected(Layer):
    weights: np.ndarray = field()
    biases: np.ndarray = field()

    input: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray):
        self.input = input
        return np.matmul(self.weights, input) + self.biases

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.matmul(output_gradient, self.input.T)
        input_gradient = np.matmul(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        push(self.weights)
        push(self.biases)

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        self.biases = pop()
        self.weights = pop()
