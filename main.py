from configuration import Configuration
from csv_manager import read_file
from neuralNetwork import NeuralNetwork


def main():
    configuration = Configuration(0.05, 20, (3072, 1000, 200, 10), 0.5, "test")
    model = NeuralNetwork(configuration)
    X, Y = read_file("train.csv")
    model.train(X, Y)


if __name__ == '__main__':
    main()

