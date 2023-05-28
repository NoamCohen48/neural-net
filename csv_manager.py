from pathlib import Path
from pprint import pprint
import numpy as np
from numpy.lib.npyio import NpzFile


def read_file(filepath) -> tuple[np.ndarray, np.ndarray]:
    # reading file to numpy array
    data = np.genfromtxt(filepath, delimiter=',', missing_values="?")
    # splitting data
    y, x = np.hsplit(data, [1])
    return x, y


def save_arrays(filepath: Path, weights: list[np.ndarray]) -> None:
    np.savez(filepath.with_suffix(""), *weights)


def load_arrays(filepath: Path) -> list[np.ndarray]:
    npz_file: NpzFile = np.load(filepath.with_suffix(".npz"))
    return [npz_file[arr] for arr in npz_file.files]


if __name__ == '__main__':
    # res = read_file("train.csv")
    save_arrays(Path("temp"), [np.arange(10), np.arange(20)])
    pprint(load_arrays("temp"))
