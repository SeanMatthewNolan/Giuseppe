from typing import Union, Optional

import casadi as ca

from giuseppe.problems.bvp.adiffInput import AdiffInputBVP
from giuseppe.problems.components.adiffInput import InputAdiffCost, InputAdiffBoundedVal


class AdiffInputOCP(AdiffInputBVP):
    """
    Class to input optimal control problem data for symbolic processing
    """

    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX):
        super().__init__(dtype=dtype)

        self.controls: InputAdiffBoundedVal = InputAdiffBoundedVal(dtype=dtype)
        self.cost: InputAdiffCost = InputAdiffCost()

    def add_control(self, var: Union[ca.SX, ca.MX],
                    lower_bound: Optional[Union[ca.SX, ca.MX, float]] = None,
                    upper_bound: Optional[Union[ca.SX, ca.MX, float]] = None):
        """
        Add a control input

        Parameters
        ----------
        var : Union[ca.SX, ca.MX]
            the independent variable (CasADi symbolic var)
        lower_bound : Optional[Union[ca.SX, ca.MX, float]]
            Minimum value of independent variable
        upper_bound : Optional[Union[ca.SX, ca.MX, float]]
            Maximum value of independent variable

        Returns
        -------
        self : AdiffInputOCP
            returns the problem object

        """
        assert(type(var) == self.dtype)
        self.controls.values = ca.vcat((self.controls.values, var))

        if lower_bound is not None:
            self.controls.lower_bound = ca.vcat((self.controls.lower_bound, lower_bound))
            self.controls.bounded = True
        else:
            self.controls.lower_bound = ca.vcat((self.controls.lower_bound, -ca.inf))

        if upper_bound is not None:
            self.controls.upper_bound = ca.vcat((self.controls.upper_bound, upper_bound))
            self.controls.bounded = True
        else:
            self.controls.upper_bound = ca.vcat((self.controls.upper_bound, ca.inf))

        return self

    def set_cost(self,
                 initial: Union[ca.SX, ca.MX, float],
                 path: Union[ca.SX, ca.MX, float],
                 terminal: Union[ca.SX, ca.MX, float]):
        self.cost = InputAdiffCost(initial, path, terminal)
        return self
