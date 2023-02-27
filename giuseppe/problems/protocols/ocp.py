from typing import Protocol, runtime_checkable, Optional

import numpy as np

from giuseppe.data_classes import Solution, Annotations


# TODO Add Vectorized Protocol
@runtime_checkable
class OCP(Protocol):
    prob_class = 'ocp'

    num_states: int
    num_controls: int
    num_parameters: int
    num_constants: int

    default_values: np.ndarray
    annotations: Optional[Annotations]

    # TODO add multi-arc support
    num_arcs: int = 1

    @staticmethod
    def compute_dynamics(
            independent: float, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_boundary_conditions(
            independent: np.ndarray, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
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
