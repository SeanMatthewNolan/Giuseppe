from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional

import numpy as np

from giuseppe.data_classes import Solution, Annotations


@runtime_checkable
class BVP(Protocol):
    prob_class = 'bvp'

    num_states: int
    num_parameters: int
    num_constants: int

    default_values: np.ndarray
    annotations: Optional[Annotations]

    # TODO add multi-arc support
    num_arcs: int = 1

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


@runtime_checkable
class BVPSeparableBC(BVP, Protocol):
    @staticmethod
    def compute_initial_boundary_conditions(
            initial_independent: float, initial_states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_terminal_boundary_conditions(
            terminal_independent: float, terminal_states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...
