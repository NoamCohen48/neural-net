from dataclasses import dataclass, field
from typing import Callable
from .layer import Layer
import numpy as np


@dataclass(slots=True)
class Softmax(Layer):
    output: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray):
        probs = np.exp(input - np.max(input, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        self.output = probs
        return probs

    def backward(self, expected):
        inp = self.output.copy()
        inp[np.arange(self.output.shape[0]), expected - 1] -= 1
        inp /= self.output.shape[0]
        return inp

    def backward_matrix(self, expected):
        return self.output - expected

    def update(self, learning_rate):
        return

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        return

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        return
