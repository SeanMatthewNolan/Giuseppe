from typing import Union

import casadi as ca

from giuseppe.problems.bvp.adiffInput import AdiffInputBVP
from giuseppe.problems.components.adiffInput import InputAdiffCost


class AdiffInputOCP(AdiffInputBVP):
    """
    Class to input optimal control problem data for symbolic processing
    """

    def __init__(self):
        super().__init__()

        self.controls = ca.MX.sym('', 0)
        self.cost: InputAdiffCost = InputAdiffCost()

    def add_control(self, var: ca.MX):
        self.controls = ca.vcat((self.controls, var))
        return self

    def set_cost(self, initial: Union[ca.MX, float], path: Union[ca.MX, float], terminal: Union[ca.MX, float]):
        self.cost = InputAdiffCost(initial, path, terminal)
        return self
