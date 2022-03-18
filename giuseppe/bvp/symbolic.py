from dataclasses import dataclass
from typing import Optional

import numpy as np
from sympy import Symbol

from giuseppe.io import InputBVP
from giuseppe.utils.mixins import Symbolic
from giuseppe.utils.typing import SymMatrix, EMPTY_SYM_MATRIX, SYM_NULL


@dataclass
class SymBoundaryConditions:
    initial: SymMatrix = EMPTY_SYM_MATRIX
    terminal: SymMatrix = EMPTY_SYM_MATRIX


class SymBVP(Symbolic):
    def __init__(self, input_data: Optional[InputBVP] = None):
        super().__init__()

        self.independent: Symbol = SYM_NULL
        self.states: SymMatrix = EMPTY_SYM_MATRIX
        self.dynamics: SymMatrix = EMPTY_SYM_MATRIX
        self.constants: SymMatrix = EMPTY_SYM_MATRIX

        self.boundary_conditions = SymBoundaryConditions()

        self.default_values = np.array([])

        if isinstance(input_data, InputBVP):
            self.process_data_from_input(input_data)

    def process_variables_from_input(self, input_data: InputBVP):
        self.independent = self.new_sym(input_data.independent)
        self.states = SymMatrix([self.new_sym(state.name) for state in input_data.states])
        self.constants = SymMatrix([self.new_sym(constant.name) for constant in input_data.constants])
        self.default_values = np.array([constant.default_value for constant in input_data.constants])

    def process_expr_from_input(self, input_data: InputBVP):
        self.dynamics = SymMatrix([self.sympify(state.eom) for state in input_data.states])
        self.boundary_conditions.initial = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.initial])
        self.boundary_conditions.terminal = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.terminal])

    def process_data_from_input(self, input_data: InputBVP):
        self.process_variables_from_input(input_data)
        self.process_expr_from_input(input_data)
