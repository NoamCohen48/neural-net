from dataclasses import dataclass
from pathlib import Path


@dataclass
class Configuration:
    learning_rate: float
    epochs: int
    layers: tuple[int]
    seed: int | None

    # @classmethod
    # def from_file(cls, path: Path) -> Self:
    #     raise NotImplemented


def configuration_from_file(path: Path) -> Configuration:
    raise NotImplemented
