from typing import Protocol, runtime_checkable

import numpy as np


# TODO Added Vectorized Protocol


@runtime_checkable
class OCP(Protocol):

    num_states: int
    num_controls: int
    num_parameters: int
    num_constants: int

    @staticmethod
    def compute_dynamics(
            independent: float, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_boundary_conditions(
            independent: tuple[np.ndarray, ...], states: tuple[np.ndarray, ...],
            parameters: np.ndarray, constants: tuple[np.ndarray, ...]
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_cost(
            independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> float:
        ...
