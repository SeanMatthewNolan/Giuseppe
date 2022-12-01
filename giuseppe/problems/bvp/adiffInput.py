from typing import Optional, Union

import casadi as ca
import numpy as np

from giuseppe.problems.components.adiffInput import InputAdiffState, InputAdiffBoundedVal, InputAdiffConstant, \
    InputAdiffConstraints, InputAdiffInequalityConstraint, InputAdiffInequalityConstraints
from giuseppe.problems.regularization import Regularizer


class AdiffInputBVP:
    """
    Class to input boundary value problem data as strings primarily for symbolic processing.
    """

    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX):
        """
        Initialize AdiffInputBVP
        """
        self.dtype = dtype
        self.independent: InputAdiffBoundedVal = InputAdiffBoundedVal(dtype=dtype)
        self.states: InputAdiffState = InputAdiffState(dtype=dtype)
        self.parameters: InputAdiffBoundedVal = InputAdiffBoundedVal(dtype=dtype)
        self.constants: InputAdiffConstant = InputAdiffConstant(dtype=dtype)
        self.constraints: InputAdiffConstraints = InputAdiffConstraints(dtype=dtype)
        self.inequality_constraints: InputAdiffInequalityConstraints = InputAdiffInequalityConstraints()

    def set_independent(self, var: Union[ca.SX, ca.MX],
                        lower_bound: Union[ca.SX, ca.MX, float] = -ca.inf,
                        upper_bound: Union[ca.SX, ca.MX, float] = ca.inf):
        """
        Set the name of the independent variable (usually time, t)

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
        self : AdiffInputBVP
            returns the problem object

        """
        assert(type(var) == self.dtype)
        self.independent.values = var
        self.independent.upper_bound = upper_bound
        self.independent.lower_bound = lower_bound
        return self

    def add_state(self, state: Union[ca.SX, ca.MX], state_eom: Union[ca.SX, ca.MX, float],
                  lower_bound: Union[ca.SX, ca.MX, float] = -ca.inf,
                  upper_bound: Union[ca.SX, ca.MX, float] = ca.inf):
        assert(type(state) == self.dtype)
        self.states.states = ca.vcat((self.states.states, state))
        self.states.eoms = ca.vcat((self.states.eoms, state_eom))
        self.states.lower_bound = ca.vcat((self.states.lower_bound, lower_bound))
        self.states.upper_bound = ca.vcat((self.states.upper_bound, upper_bound))
        return self

    def add_parameter(self, var: Union[ca.SX, ca.MX],
                      lower_bound: Union[ca.SX, ca.MX, float] = -ca.inf,
                      upper_bound: Union[ca.SX, ca.MX, float] = ca.inf):
        """
        Add a free parameter

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
        self : AdiffInputBVP
            returns the problem object

        """
        assert(type(var) == self.dtype)
        self.parameters.values = ca.vcat((self.parameters.values, var))
        self.parameters.lower_bound = ca.vcat((self.parameters.lower_bound, lower_bound))
        self.parameters.upper_bound = ca.vcat((self.parameters.upper_bound, upper_bound))
        return self

    def add_constant(self, constant: Union[ca.SX, ca.MX], default_value: Union[np.ndarray, float] = 0.0):
        self.constants.constants = ca.vcat((self.constants.constants, constant))
        self.constants.default_values = np.append(self.constants.default_values, default_value)
        return self

    def add_constraint(self, location: str, expr: Union[ca.SX, ca.MX, float]):
        """

        Parameters
        ----------
        location : str
            type of constraint: 'initial', 'terminal'
        expr : Union[ca.SX, ca.MX, float]
            expression that defines constraint

        Returns
        -------
        self : AdiffInputBVP
            returns the problem object

        """
        self.constraints.__setattr__(location, ca.vcat((self.constraints.__getattribute__(location), expr)))
        return self

    def add_inequality_constraint(
            self, location: str, expr: Union[ca.SX, ca.MX],
            lower_limit: Optional[Union[ca.SX, ca.MX, float]] = None,
            upper_limit: Optional[Union[ca.SX, ca.MX, float]] = None,
            regularizer: Optional[Regularizer] = None):
        """

        Parameters
        ----------
        location : str
            type of constraint: 'initial', 'path', 'terminal', 'control'
        expr : Union[ca.SX, ca.MX]
            expression that defines inequality constraint
        lower_limit : Optional[Union[ca.SX, ca.MX, float]]
            expression that defines lower limit of constraint
        upper_limit : Optional[Union[ca.SX, ca.MX, float]]
            expression that defines upper limit of constraint
        regularizer : Regularizer
            method for applying constraint

        Returns
        -------
        self : AdiffInputBVP
            returns the problem object

        """

        self.inequality_constraints.__getattribute__(location).append(
                InputAdiffInequalityConstraint(
                        expr, lower_limit=lower_limit, upper_limit=upper_limit, regularizer=regularizer))

        return self
