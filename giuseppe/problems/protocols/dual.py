from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Dual(Protocol):

    num_states: int
    num_controls: int
    num_parameters: int
    num_constants: int
    num_costates: int
    num_initial_adjoints: int
    num_terminal_adjoints: int
    num_adjoints: int

    @staticmethod
    def compute_costate_dynamics(
            independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_dual_boundary_conditions(
            independent: tuple[np.ndarray, ...], states: tuple[np.ndarray, ...],
            parameters: np.ndarray, constants: tuple[np.ndarray, ...]
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_hamiltonian(
            independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> float:
        ...