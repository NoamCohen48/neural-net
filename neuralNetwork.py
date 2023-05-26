import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from configuration import Configuration
from numpy.random import default_rng, Generator


@dataclass(slots=True)
class Activation:
    function: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(slots=True)
class Layer:
    weights: np.ndarray = field()
    biases: np.ndarray = field()
    input: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray):
        self.input = input
        return np.matmul(self.weights, self.input) + self.biases

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.matmul(output_gradient, self.input.T)
        input_gradient = np.matmul(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


@dataclass(slots=True)
class NeuralNetwork:
    configuration: Configuration = field()
    layers: list[np.ndarray] = field(init=False)
    random_generator: Generator = field(init=False)

    def __post_init__(self):
        self.random_generator = default_rng(self.configuration.seed)
        for input_size, output_size in itertools.pairwise(self.configuration.layers):
            self.layers.append(self.random_generator.normal(0, 0.4, (output_size, input_size)))

    def _post_processing(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplemented

    def _forward(self, input: np.ndarray):
        for layer in self.layers:
            input = np.matmul(layer, input)

    def _backward(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def predict(self):
        raise NotImplemented
