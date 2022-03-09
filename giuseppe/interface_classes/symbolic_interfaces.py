from dataclasses import dataclass
from typing import Optional

import numpy as np
from sympy import Symbol, sympify

from giuseppe.io import InputOCP
from giuseppe.utils.aliases import SymExpr, SymMatrix, EMPTY_SYM_MATRIX, SYM_NULL, SYM_ZERO


@dataclass
class SymCost:
    initial: SymExpr = SYM_ZERO
    path: SymExpr = SYM_ZERO
    terminal: SymExpr = SYM_ZERO


@dataclass
class SymBoundaryConditions:
    initial: SymMatrix = EMPTY_SYM_MATRIX
    terminal: SymMatrix = EMPTY_SYM_MATRIX


class Symbolic:
    def __init__(self):
        self.sym_locals = {}

    def new_sym(self, name: str):
        if name in self.sym_locals:
            raise ValueError(f'{name} already defined')
        elif name is None:
            raise RuntimeWarning('No varibale name given')

        sym = Symbol(name)
        self.sym_locals[name] = sym
        return sym

    def sympify(self, expr: str) -> SymExpr:
        return sympify(expr, locals=self.sym_locals)


class SymOCP(Symbolic):
    def __init__(self, input_data: Optional[InputOCP] = None):
        super().__init__()

        self.independent: Symbol = SYM_NULL
        self.states: SymMatrix = EMPTY_SYM_MATRIX
        self.dynamics: SymMatrix = EMPTY_SYM_MATRIX
        self.controls: SymMatrix = EMPTY_SYM_MATRIX
        self.constants: SymMatrix = EMPTY_SYM_MATRIX

        self.cost = SymCost()

        self.boundary_conditions = SymBoundaryConditions()

        self.default_values = np.array([])

        if isinstance(input_data, InputOCP):
            self.process_data_from_input(input_data)

    def process_data_from_input(self, input_data: InputOCP):
        self.independent = self.new_sym(input_data.independent)
        self.states = SymMatrix([self.new_sym(state.name) for state in input_data.states])
        self.controls = SymMatrix([self.new_sym(control) for control in input_data.controls])
        self.constants = SymMatrix([self.new_sym(constant.name) for constant in input_data.constants])

        self.dynamics = SymMatrix([self.sympify(state.eom) for state in input_data.states])

        self.boundary_conditions.initial = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.initial])
        self.boundary_conditions.terminal = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.terminal])

        self.default_values = np.array([constant.default_value for constant in input_data.constants])
