import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.random
from numpy.random import default_rng, Generator

from csv_manager import save_arrays, load_arrays
from Layer import Layer, FullyConnected, Loss, Activation
from configuration import Configuration
from math_functions import mean_squared_error, mean_squared_error_derivative, softmax, relu, relu_derivative


# https://github.com/TheIndependentCode/Neural-Network/blob/master/network.py#L7

@dataclass(slots=True)
class NeuralNetwork:
    configuration: Configuration = field()
    layers: list[Layer] = field(init=False, default_factory=list)
    random_generator: Generator = field(init=False)
    loss: Loss = field(init=False)

    def __post_init__(self):
        if self.configuration.seed:
            self.random_generator = default_rng(self.configuration.seed)
        else:
            seed = numpy.random.randint(100)
            print(f"using seed {seed}")
            self.random_generator = default_rng(seed)

        for input_size, output_size in itertools.pairwise(self.configuration.layers):
            self.layers.append(FullyConnected(
                self.random_generator.normal(0, 0.01, (output_size, input_size)),
                self.random_generator.normal(0, 0.01, (output_size, 1))
            ))
            self.layers.append(
                Activation(relu, relu_derivative)
            )

        self.layers.pop()

        self.loss = Loss(mean_squared_error, mean_squared_error_derivative)

    def _pre_processing(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_new = X.reshape(*X.shape, 1)
        Y_new = Y.reshape(*Y.shape, 1)
        return X_new, Y_new

    def _forward(self, input: np.ndarray):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def _backward(self, output: np.ndarray, expected: np.ndarray):
        grad = self.loss.derivative(expected, output)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update(self):
        lr = self.configuration.learning_rate
        for layer in self.layers:
            layer.update(lr)

    def train(self, train_x, train_y):
        train_x, train_y = self._pre_processing(train_x, train_y)
        for epoch in range(self.configuration.epochs):
            error = 0
            for i, (x, y) in enumerate(zip(train_x[:2000], train_y[:2000])):
                # forward
                output = self._forward(x)
                output = softmax(output)

                # calculating error
                y_true = np.zeros((self.configuration.layers[-1], 1))
                y_true[int(y) - 1] = 1
                error += self.loss.function(y_true, output)[0]

                # Backward
                self._backward(y, output)

            self._update()
            print(f"error at epoch {epoch}: {error}")
            # Save the module.
            self._save_model(epoch)

    def _save_model(self, epoch_number: int):
        path = Path(self.configuration.save_path, f"epoch{epoch_number}")
        model = []
        for layer in self.layers:
            layer.save(model.append)
        save_arrays(path, model)

    def _load_model(self, epoch_number: int | None = None):
        if epoch_number:
            path = Path(self.configuration.save_path, f"epoch{epoch_number}")
        else:
            # finding latest model in folder
            path = Path(self.configuration.save_path)
            files = path.glob("*.npz")

            def extract_number(f: Path):
                s = re.findall("\d+$", f.name)
                return int(s[0]) if s else -1, f

            epoch_number, path = max(files, key=extract_number)

        model = load_arrays(path.absolute())
        for layer in self.layers:
            layer.load(model.pop)

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
