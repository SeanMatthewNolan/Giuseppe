from dataclasses import dataclass
from typing import Optional

import numpy as np
import sympy

from giuseppe.io import InputOCP

EmptyVector = sympy.Matrix([])
SZero = sympy.Integer('0')
SymMatrix = sympy.Matrix
SymExpr = sympy.Expr


@dataclass
class SymCost:
    initial: SymExpr = SZero
    path: SymExpr = SZero
    terminal: SymExpr = SZero


@dataclass
class SymConstraints:
    initial: SymMatrix = EmptyVector
    terminal: SymMatrix = EmptyVector


class SymOCP:
    def __init__(self, input_data: Optional[InputOCP]):
        self.states: SymMatrix = EmptyVector
        self.dynamics: SymMatrix = EmptyVector
        self.controls: SymMatrix = EmptyVector
        self.constants: SymMatrix = EmptyVector

        self.cost = SymCost()

        self.constraints = SymConstraints()

        self.default_values = np.array([])

        self.sym_locals = {}

        if isinstance(input_data, InputOCP):
            self.process_data_from_input(input_data)

    def new_symbol(self, name: str):
        if name in self.sym_locals:
            raise ValueError(f'{name} already defined')

        sym = sympy.Symbol(name)
        self.sym_locals[name] = sym
        return sym

    def sympify(self, expr: str) -> sympy.Expr:
        return sympy.sympify(expr, locals=self.sym_locals)

    def process_data_from_input(self, input_data: InputOCP):
        self.states = SymMatrix([self.new_symbol(state.name) for state in input_data.states])
        self.controls = SymMatrix([self.new_symbol(control) for control in input_data.controls])
        self.constants = SymMatrix([self.new_symbol(constant.name) for constant in input_data.constants])

        self.dynamics = SymMatrix([self.sympify(state.eom) for state in input_data.states])

        self.constraints.initial = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.initial])
        self.constraints.terminal = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.terminal])

        self.default_values = np.array([constant.default_value for constant in input_data.constants])
