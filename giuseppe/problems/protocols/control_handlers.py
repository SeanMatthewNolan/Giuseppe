from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AlgebraicControlHandler(Protocol):
    @staticmethod
    def compute_control(
            independent: float, states: np.ndarray, costates: np.ndarray, parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...


@runtime_checkable
class VectorizedAlgebraicControlHandler(AlgebraicControlHandler, Protocol):
    @staticmethod
    def compute_control_vectorized(
            independent: np.ndarray, states: np.ndarray, costates: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
    ) -> np.ndarray:
        ...


@runtime_checkable
class DifferentialControlHandler(Protocol):

    @staticmethod
    def compute_control_dynamics(
            independent: float, states: np.ndarray, controls: np.ndarray, costates: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_control_boundary_conditions(
            independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...

    @staticmethod
    def compute_h_uu(
            independent: float, states: np.ndarray, controls: np.ndarray, costates: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...


@runtime_checkable
class VectorizedDifferentialControlHandler(DifferentialControlHandler, Protocol):

    @staticmethod
    def compute_control_dynamics_vectorized(
            independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
            parameters: np.ndarray, constants: np.ndarray
    ) -> np.ndarray:
        ...
