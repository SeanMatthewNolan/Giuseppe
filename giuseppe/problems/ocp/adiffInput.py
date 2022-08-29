from typing import Union

import casadi as ca

from giuseppe.problems.bvp.adiffInput import AdiffInputBVP
from giuseppe.problems.components.adiffInput import InputCost


class AdiffInputOCP(AdiffInputBVP):
    """
    Class to input optimal control problem data for symbolic processing
    """

    def __init__(self):
        super().__init__()

        self.controls = ca.SX.sym('', 0)
        self.cost: InputCost = InputCost()

    def add_control(self, var: ca.SX):
        self.controls = ca.vcat((self.controls, var))
        return self

    def set_cost(self, initial: Union[ca.SX, float], path: Union[ca.SX, float], terminal: Union[ca.SX, float]):
        self.cost = InputCost(initial, path, terminal)
        return self
