from pprint import pprint
import numpy as np
from numpy.lib.npyio import NpzFile


def read_file(filepath) -> tuple[np.ndarray, np.ndarray]:
    # reading file to numpy array
    data = np.genfromtxt(filepath, delimiter=',', missing_values="?")

    # splitting data
    # x, y = data[1:, :], data[1, :]
    x, y = np.hsplit(data, [1])

    return x, y


def save_model(filepath, weights: list[np.ndarray]) -> None:
    np.savez(filepath, *weights)


def load_model(filepath) -> list[np.ndarray]:
    npzfile: NpzFile = np.load(filepath)
    return [npzfile[arr] for arr in npzfile.files]


if __name__ == '__main__':
    # res = read_file("train.csv")
    save_model("temp", [np.arange(10), np.arange(20)])
    pprint(load_model("temp.npz"))
