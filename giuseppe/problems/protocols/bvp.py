from typing import Protocol

import numpy as np


class BVP(Protocol):
    num_states: int
    num_parameters: int
    num_constants: int

    def dynamics(
            self, independent: float, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    def boundary_conditions(
            self, independent: tuple[np.ndarray, ...], states: tuple[np.ndarray, ...],
            parameters: np.ndarray, constants: tuple[np.ndarray, ...]
    ) -> np.ndarray:
        ...


class OCP(Protocol):
    ...
