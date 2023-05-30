import os
import sys
from pathlib import Path

from configuration import Configuration
from csv_manager import read_file2
from neuralNetwork import NeuralNetwork


def main():
    print("version 1.0.1")
    configuration = Configuration(0.2, 100, (3072, 1024, 10), 41, "test")
    model = NeuralNetwork(configuration)
    train_x, train_y = read_file2(Path(sys.argv[1], "train.csv"))
    validation_x, validation_y = read_file2(Path(sys.argv[1], "validate.csv"))
    model.train(train_x, train_y, validation_x, validation_y)


if __name__ == '__main__':
    main()
