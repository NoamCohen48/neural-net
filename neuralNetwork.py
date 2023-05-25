import itertools
from dataclasses import dataclass, field
import numpy as np
from configuration import Configuration
from numpy.random import default_rng, Generator


@dataclass(slots=True)
class NeuralNetwork:
    configuration: Configuration = field()
    layers: list[np.ndarray] = field(init=False)
    generator: Generator = field(init=False)

    def __post_init__(self):
        self.generator = default_rng(self.configuration.seed)
        for input_size, output_size in itertools.pairwise(self.configuration.layers):
            self.layers.append(self.generator.normal(0, 0.4, (input_size, output_size)))

    def _post_processing(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def predict(self):
        raise NotImplemented

