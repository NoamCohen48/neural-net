from pathlib import Path
from pprint import pprint
import numpy as np
from numpy.lib.npyio import NpzFile


def read_file(filepath) -> tuple[np.ndarray, np.ndarray]:
    # reading file to numpy array
    data = np.genfromtxt(filepath, delimiter=',', missing_values="?")
    # splitting data
    x, y = np.hsplit(data, [1])
    return x, y


def save_arrays(filepath: str, weights: list[np.ndarray]) -> None:
    np.savez(filepath, *weights)


def load_arrays(filepath: str) -> list[np.ndarray]:
    path = Path(filepath).with_suffix(".npz")
    npz_file: NpzFile = np.load(path)
    return [npz_file[arr] for arr in npz_file.files]


if __name__ == '__main__':
    # res = read_file("train.csv")
    save_arrays("temp", [np.arange(10), np.arange(20)])
    pprint(load_arrays("temp"))
