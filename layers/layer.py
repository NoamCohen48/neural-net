from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
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
