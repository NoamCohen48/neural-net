import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.random
from numpy.random import default_rng, Generator

from csv_manager import save_arrays
from Layer import Layer, FullyConnected, Loss
from configuration import Configuration
from math_functions import mean_squared_error, mean_squared_error_derivative


# https://github.com/TheIndependentCode/Neural-Network/blob/master/network.py#L7

@dataclass(slots=True)
class NeuralNetwork:
    configuration: Configuration = field()
    layers: list[Layer] = field(init=False)
    random_generator: Generator = field(init=False)
    loss:Loss = field(init=False)

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

        self.loss = Loss(mean_squared_error, mean_squared_error_derivative)

    def _post_processing(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X, Y

    def _forward(self, input: np.ndarray):
        for layer in self.layers:
            output = layer.forward(input)
        return output

    def _backward(self, output: np.ndarray, expected: np.ndarray):
        lr = self.configuration.learning_rate
        grad = self.loss.derivative(expected, output)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def _save_model(self, epoch_number: int):
        path = Path(self.configuration.save_path, f"epoch{epoch_number}")
        raise NotImplemented

    def _load_model(self, epoch_number: int | None = None):
        if epoch_number:
            path = Path(self.configuration.save_path, f"epoch{epoch_number}")
        else:
            # finding latest model in folder
            path = Path(self.configuration.save_path)
            files = path.glob("*.npz")

            def extract_number(f:Path):
                s = re.findall("\d+$", f.name)
                return int(s[0]) if s else -1, f

            epoch_number, path = max(files, key=extract_number)

        raise NotImplemented

    def train(self, train_x, train_y):
        for epoch in range(self.configuration.epochs):
            error = 0
            for x, y in zip(train_x, train_y):
                # forward
                output = self._forward(x)

                error += self.loss.function(y, output)
                print(f"error at epoch {epoch}: {error}")

                # Backward
                self._backward(y, output)

                # Save the module.
                self._save_model(epoch)

    def train_by_batch(self, train_x, train_y, batch_size):
        for epoch in range(self.configuration.epochs):
            error = 0
            batch_count = 0
            batch_error = 0

            for i in range(0, len(train_x), batch_size):
                batch_x = train_x[i:i + batch_size]
                batch_y = train_y[i:i + batch_size]

                # Initialize batch error for each batch
                batch_error = 0

                for x, y in zip(batch_x, batch_y):
                    # Forward
                    output = self._forward(x)

                    # Accumulate error for the batch
                    batch_error += self.loss.function(y, output)

                    # Backward
                    self._backward(y, output)

                # Accumulate error for the epoch
                error += batch_error

                # Print batch error
                print(f"Batch error at epoch {epoch}, batch {batch_count}: {batch_error}")
                batch_count += 1

            # Print total error for the epoch
            print(f"Total error at epoch {epoch}: {error}")

            # Save the model.
            self._save_model(epoch)

