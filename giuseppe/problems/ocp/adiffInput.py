from typing import Union

import casadi as ca

from giuseppe.problems.bvp.adiffInput import AdiffInputBVP
from giuseppe.problems.components.adiffInput import InputAdiffCost, InputAdiffBoundedVal


class AdiffInputOCP(AdiffInputBVP):
    """
    Class to input optimal control problem data for symbolic processing
    """

    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX):
        super().__init__(dtype=dtype)

        self.controls = InputAdiffBoundedVal(dtype=dtype)
        self.cost: InputAdiffCost = InputAdiffCost()

    def add_control(self, var: Union[ca.SX, ca.MX],
                    lower_bound: Union[ca.SX, ca.MX, float] = -ca.inf,
                    upper_bound: Union[ca.SX, ca.MX, float] = ca.inf):
        """
        Add a control input

        Parameters
        ----------
        var : Union[ca.SX, ca.MX]
            the independent variable (CasADi symbolic var)
        lower_bound : Union[ca.SX, ca.MX, float], default=-ca.inf
            Minimum value of independent variable
        upper_bound : Union[ca.SX, ca.MX, float], default=ca.inf
            Maximum value of independent variable

        Returns
        -------
        self : AdiffInputOCP
            returns the problem object

        """
        assert(type(var) == self.dtype)
        self.controls.values = ca.vcat((self.controls.values, var))
        self.controls.lower_bound = ca.vcat((self.controls.lower_bound, lower_bound))
        self.controls.upper_bound = ca.vcat((self.controls.upper_bound, upper_bound))
        return self

    def set_cost(self,
                 initial: Union[ca.SX, ca.MX, float],
                 path: Union[ca.SX, ca.MX, float],
                 terminal: Union[ca.SX, ca.MX, float]):
        self.cost = InputAdiffCost(initial, path, terminal)
        return self
