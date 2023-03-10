from typing import Protocol, runtime_checkable, Optional

import numpy as np
from scipy.integrate import simpson

from giuseppe.data_classes import Solution, Annotations


# TODO Add Vectorized Protocol
@runtime_checkable
class OCP(Protocol):
    prob_class: str = 'ocp'

    num_states: int
    num_controls: int
    num_parameters: int
    num_constants: int

    default_values: np.ndarray
    annotations: Optional[Annotations]

    @staticmethod
    def compute_dynamics(
            independent: float, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
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
    def compute_initial_cost(
            initial_independent: float, initial_states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> float:
        ...

    @staticmethod
    def compute_terminal_cost(
            terminal_independent: float, terminal_states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> float:
        ...

    @staticmethod
    def compute_path_cost(
            independent: float, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> float:
        ...

    def compute_cost(
            self, independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> float:
        initial_cost = self.compute_initial_cost(independent[0], states[:, 0], parameters, constants)
        terminal_cost = self.compute_terminal_cost(independent[-1], states[:, -1], parameters, constants)

        instantaneous_path_costs = [
            self.compute_path_cost(ti, xi, ui, parameters, constants)
            for ti, xi, ui in zip(independent, states.T, controls.T)
        ]
        path_cost = simpson(instantaneous_path_costs, independent)

        return initial_cost + path_cost + terminal_cost

    def preprocess_data(self, data: Solution) -> Solution:
        ...

    def post_process_data(self, data: Solution) -> Solution:
        ...


@runtime_checkable
class VectorizedOCP(OCP, Protocol):
    @staticmethod
    def compute_dynamics_vectorized(
            independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_path_cost_vectorized(
            independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> np.ndarray:
        ...

    def compute_cost(
            self, independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> float:
        initial_cost = self.compute_initial_cost(independent[0], states[:, 0], parameters, constants)
        terminal_cost = self.compute_terminal_cost(independent[-1], states[:, -1], parameters, constants)

        instantaneous_path_costs = self.compute_path_cost_vectorized(
                independent, states, controls, parameters, constants)

        path_cost = simpson(instantaneous_path_costs, independent)

        return initial_cost + path_cost + terminal_cost
