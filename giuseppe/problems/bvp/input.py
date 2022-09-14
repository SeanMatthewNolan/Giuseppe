from typing import Optional

from giuseppe.data.solution import Annotations
from giuseppe.problems.components.input import InputState, InputConstant, InputNamedExpr, InputConstraints, \
    InputInequalityConstraint, InputInequalityConstraints
from giuseppe.problems.regularization import Regularizer


class InputBVP:
    """
    Class to input boundary value problem data as strings primarily for symbolic processing.
    """

    def __init__(self):
        """
        Initilize InputBVP
        """
        self.independent = None
        self.states: list[InputState] = []
        self.parameters: list[str] = []
        self.constants: list[InputConstant] = []
        self.constraints: InputConstraints = InputConstraints()
        self.inequality_constraints: InputInequalityConstraints = InputInequalityConstraints()
        self.expressions: list[InputNamedExpr] = []

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
        self.states.append(InputState(name, eom))
        return self

    def add_parameter(self, name: str):
        self.parameters.append(name)
        return self

    def add_constant(self, name: str, default_value: float = 0.):
        self.constants.append(InputConstant(name, default_value))
        return self

    def add_expression(self, name: str, expr: str):
        self.expressions.append(InputNamedExpr(name, expr))
        return self

    def add_constraint(self, location: str, expr: str):
        self.constraints.__getattribute__(location).append(expr)
        return self

    def add_inequality_constraint(
            self, location: str, expr: str, lower_limit: Optional[str] = None, upper_limit: Optional[str] = None,
            regularizer: Optional[Regularizer] = None):

        self.inequality_constraints.__getattribute__(location).append(
                InputInequalityConstraint(
                        expr, lower_limit=lower_limit, upper_limit=upper_limit, regularizer=regularizer))

        return self

    def form_annotations(self) -> Optional[Annotations]:
        return Annotations(
                t=self.independent,
                x=[state.name for state in self.states],
                p=self.parameters,
                k=[constant.name for constant in self.constants],
        )
