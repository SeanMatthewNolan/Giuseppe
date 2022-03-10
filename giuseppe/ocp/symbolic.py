from dataclasses import dataclass
from typing import Optional

from giuseppe.bvp import SymBVP
from giuseppe.io import InputOCP
from giuseppe.utils.aliases import SymExpr, SymMatrix, EMPTY_SYM_MATRIX, SYM_ZERO


@dataclass
class SymCost:
    initial: SymExpr = SYM_ZERO
    path: SymExpr = SYM_ZERO
    terminal: SymExpr = SYM_ZERO


@dataclass
class SymBoundaryConditions:
    initial: SymMatrix = EMPTY_SYM_MATRIX
    terminal: SymMatrix = EMPTY_SYM_MATRIX


class SymOCP(SymBVP):
    def __init__(self, input_data: Optional[InputOCP] = None):
        self.controls: SymMatrix = EMPTY_SYM_MATRIX
        self.cost = SymCost()

        super().__init__(input_data=input_data)

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
