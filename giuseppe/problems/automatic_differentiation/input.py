from dataclasses import dataclass
from typing import Optional, Union

import casadi as ca
import numpy as np

from giuseppe.problems.protocols import Regularizer
from giuseppe.data_classes.annotations import Annotations

from .utils import get_names


class ADiffInputState:
    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX,
                 states: Optional[Union[ca.SX, ca.MX]] = None,
                 eoms: Optional[Union[ca.SX, ca.MX]] = None):
        if states is not None:
            self.states = states
        else:
            self.states: Union[ca.SX, ca.MX] = dtype()

        if eoms is not None:
            self.eoms = eoms
        else:
            self.eoms: Union[ca.SX, ca.MX] = dtype()


class ADiffInputConstant:
    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX,
                 constants: Optional[Union[ca.SX, ca.MX]] = None,
                 default_values: Optional[np.ndarray] = None):
        if constants is not None:
            self.constants = constants
        else:
            self.constants: Union[ca.SX, ca.MX] = dtype()
        if default_values is not None:
            self.default_values = default_values
        else:
            self.default_values: np.ndarray = np.empty((0, 1))


class ADiffInputConstraints:
    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX,
                 initial: Optional[Union[ca.SX, ca.MX]] = None,
                 terminal: Optional[Union[ca.SX, ca.MX]] = None):
        if initial is not None:
            self.initial = initial
        else:
            self.initial: Union[ca.SX, ca.MX] = dtype()

        if terminal is not None:
            self.terminal = terminal
        else:
            self.terminal: Union[ca.SX, ca.MX] = dtype()


class ADiffInputInequalityConstraint:
    def __init__(self,
                 expr: Union[ca.SX, ca.MX],
                 lower_limit,
                 upper_limit,
                 regularizer: Optional[Regularizer] = None):
        self.expr: Union[ca.SX, ca.MX] = expr
        self.lower_limit: Union[ca.MX, float] = lower_limit
        self.upper_limit: Union[ca.MX, float] = upper_limit
        self.regularizer: Optional[Regularizer] = regularizer


class ADiffInputInequalityConstraints:
    def __init__(self):
        self.initial: list[ADiffInputInequalityConstraint] = []
        self.path: list[ADiffInputInequalityConstraint] = []
        self.terminal: list[ADiffInputInequalityConstraint] = []
        self.control: list[ADiffInputInequalityConstraint] = []


@dataclass
class ADiffInputCost:
    initial: Union[ca.SX, ca.MX, float] = 0.0
    path: Union[ca.SX, ca.MX, float] = 0.0
    terminal: Union[ca.SX, ca.MX, float] = 0.0


class ADiffInputProb:
    """
    Class to input boundary value problem data as strings primarily for symbolic processing.
    """

    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX):
        """
        Initialize ADiffInputProb
        """
        self.dtype = dtype
        self.independent: Union[ca.SX, ca.MX] = dtype()
        self.states: ADiffInputState = ADiffInputState(dtype=dtype)
        self.parameters: Union[ca.SX, ca.MX] = dtype()
        self.constants: ADiffInputConstant = ADiffInputConstant(dtype=dtype)
        self.constraints: ADiffInputConstraints = ADiffInputConstraints(dtype=dtype)
        self.inequality_constraints: ADiffInputInequalityConstraints = ADiffInputInequalityConstraints()

        self.controls = dtype()
        self.cost: ADiffInputCost = ADiffInputCost()

    def set_independent(self, var: Union[ca.SX, ca.MX]):
        """
        Set the name of the independent variable (usually time, t)

        Parameters
        ----------
        var : Union[ca.SX, ca.MX]
            the independent variable (CasADi symbolic var)

        Returns
        -------
        self : AdiffInputBVP
            returns the problem object

        """
        assert(type(var) == self.dtype)
        self.independent = var
        return self

    def add_state(self, state: Union[ca.SX, ca.MX], state_eom: Union[ca.SX, ca.MX, float]):
        assert(type(state) == self.dtype)
        self.states.states = ca.vcat((self.states.states, state))
        self.states.eoms = ca.vcat((self.states.eoms, state_eom))
        return self

    def add_parameter(self, var: Union[ca.SX, ca.MX]):
        assert(type(var) == self.dtype)
        self.parameters = ca.vcat((self.parameters, var))
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
        self : ADiffInputBVP
            returns the problem object

        """

        self.inequality_constraints.__getattribute__(location).append(
                ADiffInputInequalityConstraint(
                        expr, lower_limit=lower_limit, upper_limit=upper_limit, regularizer=regularizer))

        return self

    def add_control(self, var: Union[ca.SX, ca.MX]):
        assert(type(var) == self.dtype)
        self.controls = ca.vcat((self.controls, var))
        return self

    def set_cost(self,
                 initial: Union[ca.SX, ca.MX, float],
                 path: Union[ca.SX, ca.MX, float],
                 terminal: Union[ca.SX, ca.MX, float]):
        self.cost = ADiffInputCost(initial, path, terminal)
        return self

    def create_annotations(self) -> Annotations:
        annotations = Annotations(
                independent=self.independent.name(),
                states=get_names(self.states.states),
                parameters=get_names(self.parameters),
                constants=get_names(self.constants.constants)
        )

        if self.controls.numel() > 0:
            annotations.controls = get_names(self.controls)

        return annotations
