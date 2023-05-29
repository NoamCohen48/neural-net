import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.random
from numpy.random import default_rng, Generator

from csv_manager import save_arrays, load_arrays
from Layer import Layer, FullyConnected, Loss, Activation, Softmax
from configuration import Configuration
from math_functions import *


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
                self.random_generator.normal(0, 0.01, (output_size,))
            ))
            self.layers.append(
                Activation(relu, relu_derivative2)
            )

        self.layers.pop()
        self.layers.append(Softmax())

        self.loss = Loss(nll_loss, mean_squared_error_derivative)

    def _pre_processing(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # X_new = X.reshape(*X.shape, 1)
        Y_new = Y.reshape(-1).astype(int)
        return X, Y_new

    def _forward(self, input: np.ndarray):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return softmax(output)

    def _backward(self, output: np.ndarray, expected: np.ndarray):
        input = expected
        for layer in reversed(self.layers):
            input = layer.backward(input)

    def _update(self):
        lr = self.configuration.learning_rate
        for layer in self.layers:
            layer.update(lr)

    def _accuracy(self, predicted, expected):
        predicted_labels = np.argmax(predicted, axis=1)
        expected_labels = np.argmax(expected, axis=1)
        return np.mean(predicted_labels == expected_labels)

    def train(self, train_x, train_y):
        print("started training")
        train_x, train_y = self._pre_processing(train_x, train_y)
        batch_size = 200
        for epoch in range(self.configuration.epochs):
            start = epoch * batch_size
            end = start + batch_size
            batch_x, batch_y = train_x[start:end], train_y[start:end]

            # forward
            prediction = self._forward(batch_x)

            # calculating error
            expected = np.eye(self.configuration.layers[-1])[batch_y - 1]

            loss = nll_loss_matrix(expected, prediction)
            accuracy = self._accuracy(prediction, expected)

            # Backward
            self._backward(prediction, expected)
            self._update()
            print(f"#{epoch}: {loss=}, {accuracy=}")
            # Save the module.
            # self._save_model(epoch)

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
