from copy import deepcopy
from typing import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson, trapezoid

from giuseppe.problems.protocols import OCP
from giuseppe.data_classes.annotations import Annotations
from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.typing import SymMatrix, EMPTY_SYM_MATRIX, NumbaFloat, NumbaArray, NumbaMatrix, SymExpr, SYM_ZERO
from giuseppe.utils.strings import stringify_list
from .bvp import SymBVP
from .input import StrInputProb


@dataclass
class SymCost:
    initial: SymExpr = SYM_ZERO
    path: SymExpr = SYM_ZERO
    terminal: SymExpr = SYM_ZERO


class SymOCP(OCP, SymBVP):
    def __init__(self, input_data: StrInputProb = None, use_jit_compile: bool = True, cost_quadrature: str = 'simpson'):
        self.controls: SymMatrix = EMPTY_SYM_MATRIX
        self.cost = SymCost()

        if cost_quadrature.lower() == 'simpson':
            self.integrate_path_cost = simpson
        elif cost_quadrature.lower() == 'trapezoid':
            self.integrate_path_cost = trapezoid
        else:
            raise RuntimeError(f'Cannot use quadrature for cost {cost_quadrature}')

        self.compute_initial_cost = None
        self.compute_path_cost = None
        self.compute_terminal_cost = None
        self.compute_cost = None

        SymBVP.__init__(self, input_data, use_jit_compile=use_jit_compile)

    def _process_variables_from_input(self, input_data: StrInputProb):
        super()._process_variables_from_input(input_data)
        self.controls = SymMatrix([self.new_sym(control) for control in input_data.controls])

    def _process_expr_from_input(self, input_data: StrInputProb):
        super()._process_expr_from_input(input_data)
        self.cost = SymCost(
                self.sympify(input_data.cost.initial),
                self.sympify(input_data.cost.path),
                self.sympify(input_data.cost.terminal)
        )

    def _perform_substitutions(self):
        super()._perform_substitutions()
        self.cost.initial = self._substitute(self.cost.initial)
        self.cost.path = self._substitute(self.cost.path)
        self.cost.terminal = self._substitute(self.cost.terminal)

    def create_annotations(self):
        self.annotations: Annotations = Annotations(
                independent=str(self.independent),
                states=stringify_list(self.states),
                controls=stringify_list(self.controls),
                parameters=stringify_list(self.parameters),
                constants=stringify_list(self.constants),
                expressions=stringify_list([expr.sym for expr in self.expressions])
        )

        return self.annotations

    def compile_dynamics(self) -> Callable:
        _compute_dynamics = lambdify(
                self.sym_args['dynamic'], tuple(self.dynamics.flat()), use_jit_compile=self.use_jit_compile)

        def compute_dynamics(
                independent: float, states: np.ndarray, controls: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.asarray(_compute_dynamics(independent, states, controls, parameters, constants))

        if self.use_jit_compile:
            compute_dynamics = jit_compile(compute_dynamics, self.args_numba_signature['dynamic'])

        return compute_dynamics

    def compile_boundary_conditions(self) -> tuple:
        _compute_initial_boundary_conditions = lambdify(
                self.sym_args['static'], tuple(self.boundary_conditions.initial.flat()),
                use_jit_compile=self.use_jit_compile)
        _compute_terminal_boundary_conditions = lambdify(
                self.sym_args['static'], tuple(self.boundary_conditions.terminal.flat()),
                use_jit_compile=self.use_jit_compile)

        def compute_initial_boundary_conditions(
                initial_independent: float, initial_states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
        ) -> np.ndarray:
            return np.asarray(_compute_initial_boundary_conditions(
                    initial_independent, initial_states, parameters, constants))

        def compute_terminal_boundary_conditions(
                terminal_independent: float, initial_states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
        ) -> np.ndarray:
            return np.asarray(_compute_terminal_boundary_conditions(
                    terminal_independent, initial_states, parameters, constants))

        def compute_boundary_conditions(
                independent: np.ndarray, states: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:

            _psi_0 = np.asarray(
                    compute_initial_boundary_conditions(independent[0], states[:, 0], parameters, constants))
            _psi_f = np.asarray(
                    compute_terminal_boundary_conditions(independent[-1], states[:, -1], parameters, constants))

            return np.concatenate((_psi_0, _psi_f))

        if self.use_jit_compile:
            compute_initial_boundary_conditions = jit_compile(
                    compute_initial_boundary_conditions, (NumbaFloat, NumbaArray, NumbaArray, NumbaArray)
            )
            compute_terminal_boundary_conditions = jit_compile(
                    compute_terminal_boundary_conditions, (NumbaFloat, NumbaArray, NumbaArray, NumbaArray)
            )
            compute_boundary_conditions = jit_compile(
                    compute_boundary_conditions, (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray)
            )

        return compute_initial_boundary_conditions, compute_terminal_boundary_conditions, compute_boundary_conditions

    def compile_cost(self) -> tuple:
        compute_initial_cost = lambdify(
                self.sym_args['static'], self.cost.initial, use_jit_compile=self.use_jit_compile)
        compute_path_cost = lambdify(
                self.sym_args['dynamic'], self.cost.path, use_jit_compile=self.use_jit_compile)
        compute_terminal_cost = lambdify(
                self.sym_args['static'], self.cost.terminal, use_jit_compile=self.use_jit_compile)

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

    def compile(self):
        self.num_states = len(self.states)
        self.num_parameters = len(self.parameters)
        self.num_constants = len(self.constants)
        self.num_controls = len(self.controls)

        self.sym_args = {
            'dynamic': (self.independent, self.states.flat(), self.controls.flat(),
                        self.parameters.flat(), self.constants.flat()),
            'static' : (self.independent, self.states.flat(), self.parameters.flat(),
                        self.constants.flat())
        }

        self.args_numba_signature = {
            'dynamic': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
            'static' : (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
        }

        self.compute_dynamics = self.compile_dynamics()

        _boundary_condition_funcs = self.compile_boundary_conditions()
        self.compute_initial_boundary_conditions = _boundary_condition_funcs[0]
        self.compute_terminal_boundary_conditions = _boundary_condition_funcs[1]
        self.compute_boundary_conditions = _boundary_condition_funcs[2]

        _cost_funcs = self.compile_cost()
        self.compute_initial_cost = _cost_funcs[0]
        self.compute_path_cost = _cost_funcs[1]
        self.compute_terminal_cost = _cost_funcs[2]
        self.compute_cost = _cost_funcs[3]
