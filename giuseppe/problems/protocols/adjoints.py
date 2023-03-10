from typing import Protocol, runtime_checkable, Optional

import numpy as np

from giuseppe.data_classes import Annotations


@runtime_checkable
class Adjoints(Protocol):
    prob_class: str = 'adjoints'

    num_states: int
    num_controls: int
    num_parameters: int
    num_constants: int
    num_costates: int
    num_initial_adjoints: int
    num_terminal_adjoints: int
    num_adjoints: int

    default_values: np.ndarray
    annotations: Optional[Annotations]

    @staticmethod
    def compute_costate_dynamics(
            independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_initial_adjoint_boundary_conditions(
            independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, adjoints: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_terminal_adjoint_boundary_conditions(
            independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, adjoints: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_hamiltonian(
            independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> float:
        ...

    @staticmethod
    def compute_control_law(
            independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...


@runtime_checkable
class VectorizedAdjoints(Adjoints, Protocol):
    @staticmethod
    def compute_costate_dynamics_vectorized(
            independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_hamiltonian_vectorized(
            independent: np.ndarray, states: np.ndarray, controls: np.ndarray, costates: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_control_law_vectorized(
            independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...
