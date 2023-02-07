from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from giuseppe.data_classes import Solution


@runtime_checkable
class BVP(Protocol):
    num_states: int
    num_parameters: int
    num_constants: int

    default_values: np.ndarray

    @staticmethod
    def compute_dynamics(
            independent: float, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_boundary_conditions(
            independent: np.ndarray, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def preprocess_data(data: Solution) -> Solution:
        return data

    @staticmethod
    def post_process_data(data: Solution) -> Solution:
        return data
