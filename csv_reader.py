from pprint import pprint
import numpy as np


def read_file(filepath) -> tuple[np.ndarray, np.ndarray]:
    # reading file to numpy array
    data = np.genfromtxt(filepath, delimiter=',', missing_values="?")

    # splitting data
    # x, y = data[1:, :], data[1, :]
    x, y = np.hsplit(data, [1])

    return x, y





if __name__ == '__main__':
    res = read_file("train.csv")
    pprint(res)