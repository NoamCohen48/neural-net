import itertools
from dataclasses import dataclass, field

import numpy as np
import numpy.random
from numpy.random import default_rng, Generator

from Layer import Layer, FullyConnected
from configuration import Configuration


@dataclass(slots=True)
class NeuralNetwork:
    configuration: Configuration = field()
    layers: list[Layer] = field(init=False)
    random_generator: Generator = field(init=False)

    def __post_init__(self):
        if self.configuration.seed:
            self.random_generator = default_rng(self.configuration.seed)
        else:
            seed = numpy.random.random()
            print(f"using seed {seed}")
            self.random_generator = default_rng(seed)

        for input_size, output_size in itertools.pairwise(self.configuration.layers):
            self.layers.append(FullyConnected(
                self.random_generator.normal(0, 0.4, (output_size, input_size)),
                self.random_generator.normal(0, 0.4, (output_size, 1))
            ))

    def _post_processing(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplemented

    def _forward(self, input: np.ndarray):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def _backward(self, output: np.ndarray, expected: np.ndarray):
        lr = self.configuration.learning_rate
        grad = loss_prime(expected, output)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def train(self):
        raise NotImplemented

    def predict(self):
        raise NotImplemented
