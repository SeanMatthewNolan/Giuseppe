from typing import Protocol, runtime_checkable

import numpy as np

from giuseppe.data_classes import Solution


# TODO Add Vectorized Protocol


@runtime_checkable
class OCP(Protocol):

    num_states: int
    num_controls: int
    num_parameters: int
    num_constants: int

    default_values: np.ndarray

    @staticmethod
    def compute_dynamics(
            independent: float, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_boundary_conditions(
            independent: tuple[float, ...], states: tuple[np.ndarray, ...],
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_cost(
            independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> float:
        ...

    def preprocess_data(self, data: Solution) -> Solution:
        ...

    def post_process_data(self, data: Solution) -> Solution:
        ...
