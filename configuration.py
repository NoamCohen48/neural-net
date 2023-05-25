from dataclasses import dataclass


@dataclass
class Configuration:
    learning_rate: float
    epochs: int
