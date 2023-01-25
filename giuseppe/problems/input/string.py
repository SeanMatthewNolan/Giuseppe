from dataclasses import dataclass
from typing import Optional

from giuseppe.problems.regularization import Regularizer


@dataclass
class StrInputState:
    name: str
    eom: str


@dataclass
class StrInputConstant:
    name: str
    default_value: float = 0.


@dataclass
class StrInputNamedExpr:
    name: str
    expr: str


class StrInputConstraints:
    def __init__(self):
        self.initial: list[str] = []
        self.terminal: list[str] = []


@dataclass
class StrInputInequalityConstraint:
    expr: str
    lower_limit: str
    upper_limit: str
    regularizer: Optional[Regularizer] = None


class StrInputInequalityConstraints:
    def __init__(self):
        self.initial: list[StrInputInequalityConstraint] = []
        self.path: list[StrInputInequalityConstraint] = []
        self.terminal: list[StrInputInequalityConstraint] = []
        self.control: list[StrInputInequalityConstraint] = []


@dataclass
class StrInputCost:
    initial: str = '0'
    path: str = '0'
    terminal: str = '0'


class StrInputProb:
    """
    Class to input problem data as strings primarily for symbolic processing.
    """

    def __init__(self):
        """
        Initialize Problem
        """
        self.independent = None
        self.states: list[StrInputState] = []
        self.parameters: list[str] = []
        self.constants: list[StrInputConstant] = []
        self.constraints: StrInputConstraints = StrInputConstraints()
        self.inequality_constraints: StrInputInequalityConstraints = StrInputInequalityConstraints()
        self.expressions: list[StrInputNamedExpr] = []

        self.controls: list[str] = []
        self.cost: StrInputCost = StrInputCost()

    def set_independent(self, var_name: str):
        """
        Set the name of the independent variable (usually time, t)

        Parameters
        ----------
        var_name : str
            the name of the independent variable

        Returns
        -------
        self : InputBVP
            returns the problem object

        """
        self.independent = var_name
        return self

    def add_state(self, name: str, eom: str):
        self.states.append(StrInputState(name, eom))
        return self

    def add_parameter(self, name: str):
        self.parameters.append(name)
        return self

    def add_constant(self, name: str, default_value: float = 0.):
        self.constants.append(StrInputConstant(name, default_value))
        return self

    def add_expression(self, name: str, expr: str):
        self.expressions.append(StrInputNamedExpr(name, expr))
        return self

    def add_constraint(self, location: str, expr: str):
        self.constraints.__getattribute__(location).append(expr)
        return self

    def add_inequality_constraint(
            self, location: str, expr: str, lower_limit: Optional[str] = None, upper_limit: Optional[str] = None,
            regularizer: Optional[Regularizer] = None):

        self.inequality_constraints.__getattribute__(location).append(
                StrInputInequalityConstraint(
                        expr, lower_limit=lower_limit, upper_limit=upper_limit, regularizer=regularizer))

        return self

    def add_control(self, name: str):
        self.controls.append(name)
        return self

    def set_cost(self, initial: str, path: str, terminal: str):
        self.cost = StrInputCost(initial, path, terminal)
        return self
