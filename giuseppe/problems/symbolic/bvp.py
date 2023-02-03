from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
from typing import Optional, Union

import numpy as np
from sympy import Symbol, topological_sort

from giuseppe.problems.bvp.input import InputBVP
from giuseppe.problems.components.input import InputInequalityConstraints
from giuseppe.problems.input import StrInputProb
from giuseppe.problems.protocols import BVP
from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.mixins import Symbolic
from giuseppe.utils.typing import SymMatrix, EMPTY_SYM_MATRIX, SYM_NULL, SymExpr, NumbaFloat, NumbaArray, UniTuple


class SymNamedExpr:
    def __init__(self, sym):
        self.sym: Symbol = sym
        self.expr: SymExpr = sym


@dataclass
class SymBoundaryConditions:
    initial: SymMatrix = EMPTY_SYM_MATRIX
    terminal: SymMatrix = EMPTY_SYM_MATRIX


class SymBVP(Symbolic):
    def __init__(self, input_data: Optional[Union[InputBVP, StrInputProb]] = None):
        super().__init__()

        self.independent: Symbol = SYM_NULL
        self.states: SymMatrix = EMPTY_SYM_MATRIX
        self.parameters: SymMatrix = EMPTY_SYM_MATRIX
        self.dynamics: SymMatrix = EMPTY_SYM_MATRIX
        self.constants: SymMatrix = EMPTY_SYM_MATRIX

        # Holding on to these for future post-processing
        self.expressions: list[SymNamedExpr] = []

        self.boundary_conditions = SymBoundaryConditions()

        self.default_values = np.array([])

        if isinstance(input_data, InputBVP):
            self.process_data_from_input(input_data)
        elif isinstance(input_data, StrInputProb):
            self.process_data_from_input(input_data)
        elif input_data is None:
            pass
        else:
            raise RuntimeError(f'{type(self)} cannot process input data class of {type(self)}')

    def process_variables_from_input(self, input_data: InputBVP):
        self.independent = self.new_sym(input_data.independent)
        self.states = SymMatrix([self.new_sym(state.name) for state in input_data.states])
        self.parameters = SymMatrix([self.new_sym(parameter) for parameter in input_data.parameters])
        self.constants = SymMatrix([self.new_sym(constant.name) for constant in input_data.constants])

        self.default_values = np.array([constant.default_value for constant in input_data.constants])

        self.expressions = [SymNamedExpr(self.new_sym(expr.name)) for expr in input_data.expressions]

    def process_expr_from_input(self, input_data: InputBVP):
        self.dynamics = SymMatrix([self.sympify(state.eom) for state in input_data.states])
        self.boundary_conditions.initial = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.initial])
        self.boundary_conditions.terminal = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.terminal])

        for sym_expr, in_expr in zip(self.expressions, input_data.expressions):
            sym_expr.expr = self.sympify(in_expr.expr)

        self.order_named_expressions()

    def order_named_expressions(self):
        interdependencies = []
        for expr_i, expr_j in permutations(self.expressions, 2):
            if expr_i.expr.has(expr_j.sym):
                interdependencies.append((expr_i, expr_j))

        self.expressions = topological_sort((self.expressions, interdependencies), key=lambda _expr: str(_expr.sym))

    def process_inequality_constraints(self, input_inequality_constraints: InputInequalityConstraints):
        # TODO Evaluate symbolically before
        for position in ['initial', 'path', 'terminal', 'control']:
            for constraint in input_inequality_constraints.__getattribute__(position):
                if constraint.regularizer is None:
                    raise NotImplementedError('Inequality constraint without regularizer not yet implemented')
                else:
                    constraint.regularizer.apply(self, constraint, position)

    def substitute(self, sym_expr: Union[SymExpr, SymMatrix]):
        sub_pairs = [(named_expr.sym, named_expr.expr) for named_expr in self.expressions]
        return sym_expr.subs(sub_pairs)

    def perform_substitutions(self):
        self.dynamics = self.substitute(self.dynamics)
        self.boundary_conditions.initial = self.substitute(self.boundary_conditions.initial)
        self.boundary_conditions.terminal = self.substitute(self.boundary_conditions.terminal)

    def process_data_from_input(self, input_data: Union[InputBVP, StrInputProb]):
        self.process_variables_from_input(input_data)
        self.process_expr_from_input(input_data)
        self.process_inequality_constraints(input_data.inequality_constraints)
        self.perform_substitutions()

    def compile(self, use_jit_compile: bool = True) -> 'CompBVP':
        return CompBVP(self, use_jit_compile=use_jit_compile)


class CompBVP(BVP):
    def __init__(self, source_bvp: SymBVP, use_jit_compile: bool = True):
        self.use_jit_compile = use_jit_compile
        self.source_bvp = deepcopy(source_bvp)

        self.num_states = len(self.source_bvp.states)
        self.num_parameters = len(self.source_bvp.parameters)
        self.num_constants = len(self.source_bvp.constants)
        self.default_values = self.source_bvp.default_values

        self.sym_args = (self.source_bvp.independent, self.source_bvp.states.flat(), self.source_bvp.parameters.flat(),
                         self.source_bvp.constants.flat())
        self.args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray)

        self.compute_dynamics = self.compile_dynamics()

        _boundary_condition_funcs = self.compile_boundary_conditions()
        self.compute_initial_boundary_conditions = _boundary_condition_funcs[0]
        self.compute_terminal_boundary_conditions = _boundary_condition_funcs[1]
        self.compute_boundary_conditions = _boundary_condition_funcs[2]

    def compile_dynamics(self):
        _compute_dynamics = lambdify(self.sym_args, self.source_bvp.dynamics.flat(),
                                     use_jit_compile=self.use_jit_compile)

        def compute_dynamics(
                independent: float, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.asarray(_compute_dynamics(independent, states, parameters, constants))

        if self.use_jit_compile:
            compute_dynamics = jit_compile(compute_dynamics, self.args_numba_signature)

        return compute_dynamics

    def compile_boundary_conditions(self):
        compute_initial_boundary_conditions = lambdify(
                self.sym_args, tuple(self.source_bvp.boundary_conditions.initial.flat()),
                use_jit_compile=self.use_jit_compile)
        compute_terminal_boundary_conditions = lambdify(
                self.sym_args, tuple(self.source_bvp.boundary_conditions.terminal.flat()),
                use_jit_compile=self.use_jit_compile)

        def compute_boundary_conditions(
                independent: tuple[float, float], states: tuple[np.ndarray, np.ndarray],
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            _bc_0 = np.asarray(compute_initial_boundary_conditions(independent[0], states[0], parameters, constants))
            _bc_f = np.asarray(compute_terminal_boundary_conditions(independent[1], states[1], parameters, constants))

            return np.concatenate((_bc_0, _bc_f))

        if self.use_jit_compile:
            compute_boundary_conditions = jit_compile(
                    compute_boundary_conditions,
                    (UniTuple(NumbaFloat, 2), UniTuple(NumbaArray, 2), NumbaArray, NumbaArray)
            )

        return compute_initial_boundary_conditions, compute_terminal_boundary_conditions, compute_boundary_conditions
