from typing import Optional

from giuseppe.problems.bvp import SymBVP
from giuseppe.problems.components.symbolic import SymCost
from giuseppe.problems.ocp.input import InputOCP
from giuseppe.utils.typing import SymMatrix, EMPTY_SYM_MATRIX


class SymOCP(SymBVP):
    def __init__(self, input_data: Optional[InputOCP] = None):
        self.controls: SymMatrix = EMPTY_SYM_MATRIX
        self.cost = SymCost()

        super().__init__(input_data=input_data)

        self.num_states = len(self.states)
        self.num_parameters = len(self.parameters)
        self.num_constants = len(self.constants)
        self.num_controls = len(self.controls)

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
