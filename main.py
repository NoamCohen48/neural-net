import os
import sys

from configuration import Configuration
from csv_manager import read_file2
from neuralNetwork import NeuralNetwork

def main():
    print("version 1.0.0")
    configuration = Configuration(0.001, 300, (3072, 1024, 512, 10), 12, "test")
    model = NeuralNetwork(configuration)
    X, Y = read_file2(sys.argv[1])
    model.train(X, Y)


if __name__ == '__main__':
    main()

