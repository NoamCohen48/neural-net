import os

from configuration import Configuration
from csv_manager import read_file2
from neuralNetwork import NeuralNetwork

def main():
    configuration = Configuration(0.001, 100, (3072, 1024, 512, 10), 12, "test")
    model = NeuralNetwork(configuration)
    print(os.listdir())
    # X, Y = read_file2("data/train.csv")
    # model.train(X, Y)


if __name__ == '__main__':
    main()

