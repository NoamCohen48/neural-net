from configuration import Configuration
from csv_manager import read_file2
from neuralNetwork import NeuralNetwork


def main():
    configuration = Configuration(0.1, 20, (3072, 1024, 512, 10), 12, "test")
    model = NeuralNetwork(configuration)
    X, Y = read_file2("data/train.csv")
    model.train(X, Y)


if __name__ == '__main__':
    main()

