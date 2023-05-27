from dataclasses import dataclass
from pathlib import Path


@dataclass
class Configuration:
    learning_rate: float
    epochs: int
    layers: list[int]
    seed: int | None
    save_path:str

    # @classmethod
    # def from_file(cls, path: Path) -> Self:
    #     raise NotImplemented


def configuration_from_file(path: Path) -> Configuration:
    raise NotImplemented
