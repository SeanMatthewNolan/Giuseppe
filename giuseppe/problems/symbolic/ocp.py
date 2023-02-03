from copy import deepcopy
from typing import Optional, Union, Callable

import numpy as np
from scipy.integrate import simpson, trapezoid

from giuseppe.problems.components.symbolic import SymCost
from giuseppe.problems.input import StrInputProb
from giuseppe.problems.ocp.input import InputOCP
from giuseppe.problems.protocols import OCP
from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.typing import SymMatrix, EMPTY_SYM_MATRIX, NumbaFloat, NumbaArray, UniTuple

from .bvp import SymBVP


class SymOCP(SymBVP):
    def __init__(self, input_data: Optional[Union[InputOCP, StrInputProb]] = None):
        self.controls: SymMatrix = EMPTY_SYM_MATRIX
        self.cost = SymCost()

        super().__init__(input_data=input_data)

        self.num_states = len(self.states)
        self.num_parameters = len(self.parameters)
        self.num_constants = len(self.constants)
        self.num_controls = len(self.controls)
        self.sym_args = (self.independent, self.states.flat(), self.controls.flat(),
                         self.parameters.flat(), self.constants.flat())

    def process_variables_from_input(self, input_data: InputOCP):
        super().process_variables_from_input(input_data)
        self.controls = SymMatrix([self.new_sym(control) for control in input_data.controls])

    def process_expr_from_input(self, input_data: InputOCP):
        super().process_expr_from_input(input_data)
        self.cost = SymCost(
                self.sympify(input_data.cost.initial),
                self.sympify(input_data.cost.path),
                self.sympify(input_data.cost.terminal)
        )

    def perform_substitutions(self):
        super().perform_substitutions()
        self.cost.initial = self.substitute(self.cost.initial)
        self.cost.path = self.substitute(self.cost.path)
        self.cost.terminal = self.substitute(self.cost.terminal)


class CompOCP(OCP):
    def __init__(self, source_ocp: SymOCP, use_jit_compile: bool = True, cost_quadrature: str = 'simpson'):
        self.use_jit_compile = use_jit_compile
        self.source_ocp: SymOCP = deepcopy(source_ocp)

        self.num_states: int = len(self.source_ocp.states)
        self.num_parameters: int = len(self.source_ocp.parameters)
        self.num_constants: int = len(self.source_ocp.constants)
        self.default_values: np.ndarray = self.source_ocp.default_values

        self.sym_args = {
            'dynamic': (self.source_ocp.independent, self.source_ocp.states.flat(), self.source_ocp.controls.flat(),
                        self.source_ocp.parameters.flat(), self.source_ocp.constants.flat()),
            'static':  (self.source_ocp.independent, self.source_ocp.states.flat(), self.source_ocp.parameters.flat(),
                        self.source_ocp.constants.flat())
        }

        self.args_numba_signature = {
            'dynamic': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
            'static':  (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
        }

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
        _compute_dynamics = lambdify(
                self.sym_args['dynamic'], self.source_ocp.dynamics.flat(), use_jit_compile=self.use_jit_compile)

        def compute_dynamics(
                independent: float, states: np.ndarray, controls: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.array(_compute_dynamics(independent, states, controls, parameters, constants))

        if self.use_jit_compile:
            compute_dynamics = jit_compile(compute_dynamics, self.args_numba_signature['dynamic'])

        return compute_dynamics

    def compile_boundary_conditions(self) -> tuple:
        compute_initial_boundary_conditions = lambdify(
                self.sym_args['static'], tuple(self.source_ocp.boundary_conditions.initial.flat()),
                use_jit_compile=self.use_jit_compile)
        compute_terminal_boundary_conditions = lambdify(
                self.sym_args['static'], tuple(self.source_ocp.boundary_conditions.terminal.flat()),
                use_jit_compile=self.use_jit_compile)

        def compute_boundary_conditions(
                independent: tuple[float, float], states: tuple[np.ndarray, np.ndarray],
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:

            _psi_0 = np.array(compute_initial_boundary_conditions(independent[0], states[0], parameters, constants))
            _psi_f = np.array(compute_terminal_boundary_conditions(independent[1], states[1], parameters, constants))

            return np.concatenate((_psi_0, _psi_f))

        if self.use_jit_compile:
            compute_boundary_conditions = jit_compile(
                    compute_boundary_conditions,
                    (UniTuple(NumbaFloat, 2), UniTuple(NumbaArray, 2), NumbaArray, NumbaArray)
            )

        return compute_initial_boundary_conditions, compute_terminal_boundary_conditions, compute_boundary_conditions

    def compile_cost(self) -> tuple:
        compute_initial_cost = lambdify(
               self.sym_args['static'], self.source_ocp.cost.initial, use_jit_compile=self.use_jit_compile)
        compute_path_cost = lambdify(
               self.sym_args['dynamic'], self.source_ocp.cost.path, use_jit_compile=self.use_jit_compile)
        compute_terminal_cost = lambdify(
                self.sym_args['static'], self.source_ocp.cost.terminal, use_jit_compile=self.use_jit_compile)

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
