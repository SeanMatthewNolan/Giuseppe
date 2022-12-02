from dataclasses import dataclass
from typing import Optional, Union

import casadi as ca
import numpy as np

from giuseppe.problems.regularization import Regularizer


class InputAdiffState:
    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX,
                 states: Optional[Union[ca.SX, ca.MX]] = None,
                 eoms: Optional[Union[ca.SX, ca.MX]] = None,
                 upper_bound: Optional[Union[ca.SX, ca.MX]] = None,
                 lower_bound: Optional[Union[ca.SX, ca.MX]] = None):

        self.bounded = False

        if states is not None:
            self.states = states
        else:
            self.states: Union[ca.SX, ca.MX] = dtype()

        if eoms is not None:
            self.eoms = eoms
        else:
            self.eoms: Union[ca.SX, ca.MX] = dtype()

        if upper_bound is not None:
            self.upper_bound = upper_bound
            self.bounded = True
        else:
            self.upper_bound = dtype()

        if lower_bound is not None:
            self.lower_bound = lower_bound
            self.bounded = True
        else:
            self.lower_bound = dtype()


class InputAdiffBoundedVal:
    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX,
                 values: Optional[Union[ca.SX, ca.MX]] = None,
                 upper_bound: Optional[Union[ca.SX, ca.MX]] = None,
                 lower_bound: Optional[Union[ca.SX, ca.MX]] = None):

        self.bounded = False

        if values is not None:
            self.values = values
        else:
            self.values: Union[ca.SX, ca.MX] = dtype()

        if upper_bound is not None:
            self.upper_bound = upper_bound
            self.bounded = True
        else:
            self.upper_bound = dtype()

        if lower_bound is not None:
            self.lower_bound = lower_bound
            self.bounded = True
        else:
            self.lower_bound = dtype()


class InputAdiffIndependent(InputAdiffBoundedVal):
    def __init__(self, dtype: Union[type(ca.SX), type(ca.MX)] = ca.SX,
                 values: Optional[Union[ca.SX, ca.MX]] = None,
                 upper_bound: Optional[Union[ca.SX, ca.MX]] = None,
                 lower_bound: Optional[Union[ca.SX, ca.MX]] = None,
                 increasing: Optional[bool] = None):
        super().__init__(dtype, values, upper_bound, lower_bound)
        self.increasing = increasing


class InputAdiffConstant:
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


class InputAdiffConstraints:
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


class InputAdiffInequalityConstraint:
    def __init__(self,
                 expr: Union[ca.SX, ca.MX],
                 lower_limit,
                 upper_limit,
                 regularizer: Optional[Regularizer] = None):
        self.expr: Union[ca.SX, ca.MX] = expr
        self.lower_limit: Union[ca.MX, float] = lower_limit
        self.upper_limit: Union[ca.MX, float] = upper_limit
        self.regularizer: Optional[Regularizer] = regularizer


class InputAdiffInequalityConstraints:
    def __init__(self):
        self.initial: list[InputAdiffInequalityConstraint] = []
        self.path: list[InputAdiffInequalityConstraint] = []
        self.terminal: list[InputAdiffInequalityConstraint] = []
        self.control: list[InputAdiffInequalityConstraint] = []


@dataclass
class InputAdiffCost:
    initial: Union[ca.SX, ca.MX, float] = 0.0
    path: Union[ca.SX, ca.MX, float] = 0.0
    terminal: Union[ca.SX, ca.MX, float] = 0.0
