from copy import deepcopy
from typing import Callable

import numpy as np
from scipy.integrate import simpson, trapezoid

from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.typing import NumbaFloat, NumbaArray, UniTuple
from ...protocols import OCP
from ..intermediate import SymOCP


# TODO add interior point and mult-arc support
class CompOCP(OCP):
    def __init__(self, source_ocp: SymOCP, use_jit_compile: bool = True, cost_quadrature: str = 'simpson'):
        self.use_jit_compile = use_jit_compile
        self.source_ocp: SymOCP = deepcopy(source_ocp)

        self.num_states: int = len(self.source_ocp.states)
        self.num_parameters: int = len(self.source_ocp.parameters)
        self.num_constants: int = len(self.source_ocp.constants)
        self.default_values: np.ndarray = self.source_ocp.default_values

        self.dyn_sym_args = (
            self.source_ocp.independent, self.source_ocp.states.flat(), self.source_ocp.controls.flat(),
            self.source_ocp.parameters.flat(), self.source_ocp.constants.flat())
        self.dyn_args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray)

        self.bc_sym_args = (
            self.source_ocp.independent, self.source_ocp.states.flat(), self.source_ocp.parameters.flat(),
            self.source_ocp.constants.flat())
        self.bc_args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray)

        self.compute_dynamics = self.compile_dynamics()

        _boundary_condition_funcs = self.compile_boundary_conditions()
        self.compute_initial_boundary_conditions = _boundary_condition_funcs[0]
        self.compute_terminal_boundary_conditions = _boundary_condition_funcs[1]
        self.compute_boundary_conditions = _boundary_condition_funcs[2]

        if cost_quadrature.lower() == 'simpson':
            self.integrate_path_cost = simpson
        elif cost_quadrature.lower() == 'trapezoid':
            self.integrate_path_cost = trapezoid

        _cost_funcs = self.compile_cost()
        self.compute_initial_cost = _cost_funcs[0]
        self.compute_path_cost = _cost_funcs[1]
        self.compute_terminal_cost = _cost_funcs[2]
        self.compute_cost = _cost_funcs[3]

    def compile_dynamics(self) -> Callable:
        _dyn = lambdify(self.dyn_sym_args, self.source_ocp.dynamics.flat(), use_jit_compile=self.use_jit_compile)

        def compute_dynamics(
                independent: float, states: np.ndarray, controls: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.array(_dyn(independent, states, controls, parameters, constants))

        if self.use_jit_compile:
            compute_dynamics = jit_compile(compute_dynamics, self.dyn_args_numba_signature)

        return compute_dynamics

    def compile_boundary_conditions(self) -> tuple:
        compute_initial_boundary_conditions = lambdify(
                self.bc_sym_args, tuple(self.source_ocp.boundary_conditions.initial.flat()),
                use_jit_compile=self.use_jit_compile)
        compute_terminal_boundary_conditions = lambdify(
                self.bc_sym_args, tuple(self.source_ocp.boundary_conditions.terminal.flat()),
                use_jit_compile=self.use_jit_compile)

        def compute_boundary_conditions(
                independent: tuple[float, float], states: tuple[np.ndarray, np.ndarray],
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:

            _bc_0 = np.array(compute_initial_boundary_conditions(independent[0], states[0], parameters, constants))
            _bc_f = np.array(compute_terminal_boundary_conditions(independent[1], states[1], parameters, constants))

            return np.concatenate((_bc_0, _bc_f))

        if self.use_jit_compile:
            compute_boundary_conditions = jit_compile(
                    compute_boundary_conditions,
                    (UniTuple(NumbaFloat, 2), UniTuple(NumbaArray, 2), NumbaArray, NumbaArray)
            )

        return compute_initial_boundary_conditions, compute_terminal_boundary_conditions, compute_boundary_conditions

    def compile_cost(self) -> tuple:
        compute_initial_cost = lambdify(
               self.bc_sym_args, self.source_ocp.cost.initial, use_jit_compile=self.use_jit_compile)
        compute_path_cost = lambdify(
               self.dyn_sym_args, self.source_ocp.cost.path, use_jit_compile=self.use_jit_compile)
        compute_terminal_cost = lambdify(
                self.bc_sym_args, self.source_ocp.cost.terminal, use_jit_compile=self.use_jit_compile)

        integrate_path_cost = self.integrate_path_cost

        def compute_cost(
            independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
            constants: np.ndarray
        ) -> float:
            initial_cost = compute_initial_cost(independent[0], states[:, 0], parameters, constants)
            terminal_cost = compute_terminal_cost(independent[-1], states[:, -1], parameters, constants)

            instantaneous_path_costs = [
                compute_path_cost(ti, xi, ui, parameters, constants)
                for ti, xi, ui in zip(independent, states.T, controls.T)
            ]
            path_cost = integrate_path_cost(instantaneous_path_costs, independent)

            return initial_cost + path_cost + terminal_cost

        return compute_initial_cost, compute_path_cost, compute_terminal_cost, compute_cost
