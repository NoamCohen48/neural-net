import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.random
from numpy.random import default_rng, Generator

from csv_manager import save_arrays, load_arrays
from layers import *
from configuration import Configuration
from math_functions import *


# https://github.com/TheIndependentCode/Neural-Network/blob/master/network.py#L7

@dataclass(slots=True)
class NeuralNetwork:
    configuration: Configuration = field()
    layers: list[Layer] = field(init=False, default_factory=list)
    random_generator: Generator = field(init=False)

    def __post_init__(self):
        if self.configuration.seed:
            self.random_generator = default_rng(self.configuration.seed)
        else:
            seed = numpy.random.randint(100)
            print(f"using seed {seed}")
            self.random_generator = default_rng(seed)

        for input_size, output_size in itertools.pairwise(self.configuration.layers):
            self.layers.append(FullyConnected(input_size, output_size, self.random_generator))
            self.layers.append(
                ReLU()
            )

        self.layers.pop()
        self.layers.append(Softmax())

    def _pre_processing(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # X_new = X.reshape(*X.shape, 1)
        Y_new = Y.reshape(-1).astype(int)
        return X, Y_new

    def _forward(self, input: np.ndarray):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

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
        # print(f"predicted={predicted_labels}")
        return np.sum(predicted_labels == expected)

    def train(self, train_x, train_y, validation_x, validation_y):
        print("started training")
        train_x, train_y = self._pre_processing(train_x, train_y)
        validation_x, validation_y = self._pre_processing(validation_x, validation_y)
        batch_size = 200
        evaluation_every = 2
        learning_rate_reduction = 0.8
        learning_rate = self.configuration.learning_rate
        for epoch in range(self.configuration.epochs):
            epoch_loss, epoch_accuracy = 0, 0
            for batch_index, bach_start in enumerate(range(0, train_x.shape[0], batch_size)):
                # ---- creating batch ----
                batch_end = bach_start + batch_size
                batch_x, batch_y = train_x[bach_start:batch_end], train_y[bach_start:batch_end]

                # ---- forward ----
                prediction = batch_x
                for layer in self.layers:
                    prediction = layer.forward(prediction)

                # ---- error ----
                batch_loss = nll_loss(batch_y, prediction)
                batch_accuracy = self._accuracy(prediction, batch_y)
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy

                # ---- backward ----
                grad = batch_y
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                for layer in reversed(self.layers):
                    layer.update(learning_rate)

                print(f"-epoch #{epoch}, batch #{batch_index}\n\tbatch loss={batch_loss / batch_size}\n\tbatch accuracy={batch_accuracy / batch_size * 100}")

            print(f"-epoch:{epoch}\n\tepoch loss={epoch_loss / train_x.shape[0]}\n\tbatch accuracy={epoch_accuracy / train_x.shape[0] * 100}")
            learning_rate *= learning_rate_reduction
            # Save the module.
            # self._save_model(epoch)

            if epoch % evaluation_every == 0:
                # ---- forward ----
                prediction = validation_x
                for layer in self.layers:
                    prediction = layer.forward(prediction)

                # ---- error ----
                loss = nll_loss(validation_y, prediction)
                accuracy = self._accuracy(prediction, batch_y)
                print(f"-validation\nloss={loss}\naccuracy={accuracy}")

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
