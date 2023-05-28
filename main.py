from configuration import Configuration
from csv_manager import read_file, read_file2
from neuralNetwork import NeuralNetwork


def main():
    configuration = Configuration(0.05, 20, (3072, 1000, 200, 10), 30, "test")
    model = NeuralNetwork(configuration)
    X, Y = read_file2("data/test.csv")
    model.train(X, Y)


if __name__ == '__main__':
    main()

