from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BVP(Protocol):
    num_states: int
    num_parameters: int
    num_constants: int

    default_values: np.ndarray

    @staticmethod
    def dynamics(
            independent: float, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def boundary_conditions(
            independent: tuple[float, ...], states: tuple[np.ndarray, ...],
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...
