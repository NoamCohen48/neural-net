from dataclasses import dataclass, field, InitVar
from typing import Callable

from numpy.random import Generator

from .layer import Layer
import numpy as np


@dataclass(slots=True)
class FullyConnected(Layer):
    input_size: InitVar[int]
    output_size: InitVar[int]
    random_generator: InitVar[Generator]

    weights: np.ndarray = field(init=False)
    biases: np.ndarray = field(init=False)

    weights_momentum: np.ndarray = field(init=False)
    biases_momentum: np.ndarray = field(init=False)

    weights_gradient: np.ndarray = field(init=False)
    biases_gradient: np.ndarray = field(init=False)

    input: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self, input_size: int, output_size: int, random_generator: Generator):
        self.weights = random_generator.standard_normal((input_size, output_size)) / np.sqrt(input_size / 2)
        self.biases = np.zeros((output_size,))

        self.weights_momentum = np.zeros_like(self.weights)
        self.biases_momentum = np.zeros_like(self.biases)

        self._reset_gradients()

    def _reset_gradients(self):
        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

    def forward(self, input: np.ndarray):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient):
        self.weights_gradient += np.dot(self.input.T, output_gradient)
        self.biases_gradient += np.sum(output_gradient, axis=0)
        return np.dot(output_gradient, self.weights.T)

    def update(self, learning_rate):
        self.weights *= (1 - 4e-4)
        self.biases *= (1 - 4e-4)

        self.weights_momentum = 0.9 * self.weights_momentum - learning_rate * self.weights_gradient
        self.biases_momentum = 0.9 * self.biases_momentum - learning_rate * self.biases_gradient

        self.weights += self.weights_momentum
        self.biases += self.biases_momentum

        self._reset_gradients()

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        push(self.weights)
        push(self.biases)

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        self.biases = pop()
        self.weights = pop()
