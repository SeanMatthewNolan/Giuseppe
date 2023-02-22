from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Adjoints(Protocol):
    prob_class = 'adjoints'

    num_states: int
    num_controls: int
    num_parameters: int
    num_constants: int
    num_costates: int
    num_initial_adjoints: int
    num_terminal_adjoints: int
    num_adjoints: int

    # TODO add multi-arc support
    num_arcs: int = 1

    @staticmethod
    def compute_costate_dynamics(
            independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_adjoint_boundary_conditions(
            independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
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
