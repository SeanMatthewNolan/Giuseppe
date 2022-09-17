from typing import Union

import casadi as ca

from giuseppe.problems.bvp.adiffInput import AdiffInputBVP
from giuseppe.problems.components.adiffInput import InputAdiffCost


class AdiffInputOCP(AdiffInputBVP):
    """
    Class to input optimal control problem data for symbolic processing
    """

    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX):
        super().__init__(dtype=dtype)

        self.controls = dtype()
        self.cost: InputAdiffCost = InputAdiffCost()

    def add_control(self, var: Union[ca.SX, ca.MX]):
        assert(type(var) == self.dtype)
        self.controls = ca.vcat((self.controls, var))
        return self

    def set_cost(self,
                 initial: Union[ca.SX, ca.MX, float],
                 path: Union[ca.SX, ca.MX, float],
                 terminal: Union[ca.SX, ca.MX, float]):
        self.cost = InputAdiffCost(initial, path, terminal)
        return self
