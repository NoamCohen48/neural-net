from dataclasses import dataclass, field
from typing import Callable

from math_functions import relu_derivative2, relu3
from .layer import Layer

import numpy as np


@dataclass(slots=True)
class ReLU(Layer):
    input: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray):
        self.input = input
        return relu3(input)

    def backward(self, output_gradient):
        return relu_derivative2(self.input, output_gradient)

    def update(self, learning_rate):
        return

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        return

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        return
