from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional, Callable

import numpy as np

from giuseppe.data_classes import Solution, Annotations


_process_type = Callable[['Problem', Solution], Solution]


@runtime_checkable
class BVP(Protocol):
    prob_class: str = 'bvp'

    num_states: int
    num_parameters: int
    num_constants: int

    preprocesses: list[_process_type] = []
    post_processes: list[_process_type] = []

    default_values: np.ndarray
    annotations: Optional[Annotations]

    @staticmethod
    def compute_dynamics(
            independent: float, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

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

    @staticmethod
    def compute_cost(
            independent: np.ndarray, states: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> float:
        ...

    def preprocess_data(self, data: Solution) -> Solution:
        for process in self.preprocesses:
            data = process(self, data)
        return data

    def post_process_data(self, data: Solution) -> Solution:
        for process in self.post_processes:
            data = process(self, data)
        return data


@runtime_checkable
class VectorizedBVP(BVP, Protocol):
    @staticmethod
    def compute_dynamics_vectorized(
            independent: np.ndarray, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...
