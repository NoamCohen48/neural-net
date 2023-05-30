from dataclasses import dataclass, field, InitVar
from typing import Callable

import numpy as np
from numpy.random import Generator

import math_functions








@dataclass(slots=True)
class Loss:
    function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]





