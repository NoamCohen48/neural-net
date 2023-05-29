import os
import sys

from configuration import Configuration
from csv_manager import read_file2
from neuralNetwork import NeuralNetwork

def main():
    print("version 1.0.1")
    configuration = Configuration(0.1, 100, (3072, 1024, 10), None, "test")
    model = NeuralNetwork(configuration)
    X, Y = read_file2(sys.argv[1])
    model.train(X, Y)


if __name__ == '__main__':
    main()

