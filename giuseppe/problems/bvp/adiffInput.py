from typing import Optional, Union

import casadi as ca

from giuseppe.problems.components.adiffInput import InputAdiffState, InputAdiffConstant,\
    InputAdiffConstraints, InputAdiffInequalityConstraint, InputAdiffInequalityConstraints
from giuseppe.problems.regularization import Regularizer


class AdiffInputBVP:
    """
    Class to input boundary value problem data as strings primarily for symbolic processing.
    """

    def __init__(self):
        """
        Initilize AdiffInputBVP
        """
        self.independent = None
        self.states: InputAdiffState = InputAdiffState()
        self.parameters: ca.SX = ca.SX.sym('', 0)
        self.constants: InputAdiffConstant = InputAdiffConstant()
        self.constraints: InputAdiffConstraints = InputAdiffConstraints()
        self.inequality_constraints: InputAdiffInequalityConstraints = InputAdiffInequalityConstraints()

    def set_independent(self, var: ca.SX):
        """
        Set the name of the independent variable (usually time, t)

        Parameters
        ----------
        var : ca.SX
            the independent variable (CasADi symbolic var)

        Returns
        -------
        self : AdiffInputBVP
            returns the problem object

        """
        self.independent = var
        return self

    def add_state(self, state: ca.SX, state_eom: Union[ca.SX, float]):
        self.states.states = ca.vcat((self.states.states, state))
        self.states.eoms = ca.vcat((self.states.eoms, state_eom))
        return self

    def add_parameter(self, var: ca.SX):
        self.parameters.append(var)
        return self

    def add_constant(self, constant: ca.SX, default_value: Union[ca.SX, float] = ca.SX(0)):
        self.constants.constants = ca.vcat((self.constants.constants, constant))
        self.constants.default_values = ca.vcat((self.constants.default_values, default_value))
        return self

    def add_constraint(self, location: str, expr: Union[ca.SX, float]):
        """

        Parameters
        ----------
        location : str
            type of constraint: 'initial', 'terminal'
        expr : ca.SX
            expression that defines constraint

        Returns
        -------
        self : AdiffInputBVP
            returns the problem object

        """
        self.constraints.__setattr__(location, ca.vcat((self.constraints.__getattribute__(location), expr)))
        return self

    def add_inequality_constraint(
            self, location: str, expr: ca.SX,
            lower_limit: Optional[Union[ca.SX, float]] = None, upper_limit: Optional[Union[ca.SX, float]] = None,
            regularizer: Optional[Regularizer] = None):
        """

        Parameters
        ----------
        location : str
            type of constraint: 'initial', 'path', 'terminal', 'control'
        expr : ca.SX
            expression that defines inequality constraint
        lower_limit : ca.SX
            expression that defines lower limit of constraint
        upper_limit : ca.SX
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
