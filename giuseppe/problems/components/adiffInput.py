from dataclasses import dataclass
from typing import Optional, Union

import casadi as ca
import numpy as np

from giuseppe.problems.regularization import Regularizer


@dataclass
class InputAdiffState:
    states: ca.SX = ca.SX.sym('', 0)
    eoms: ca.SX = ca.SX.sym('', 0)


@dataclass
class InputAdiffConstant:
    constants: ca.SX = ca.SX.sym('', 0)
    default_values: np.ndarray = np.empty((0, 1))


class InputAdiffNamedExpr:
    def __init__(self, var: ca.SX, func: ca.Function):
        self.var: ca.SX = var
        self.func: ca.Function = func
        self.wrapped_func = self.wrap_expr()

    def wrap_expr(self):
        default_func = ca.Function(self.var.str(), self.func.sx_in(), (np.zeros(self.func.sx_out()[0].shape),))

        def wrapped_func(*args):
            try:
                return self.func(*args)
            except RuntimeError:
                return default_func(*args)

        return wrapped_func

    def __repr__(self):
        return f'NamedExpr {self.var.str()} for {self.func.str()}'


@dataclass
class InputAdiffConstraints:
    initial: ca.SX = ca.SX.sym('', 0)
    terminal: ca.SX = ca.SX.sym('', 0)


@dataclass
class InputAdiffInequalityConstraint:
    expr: ca.SX
    lower_limit: Union[ca.SX, float]
    upper_limit: Union[ca.SX, float]
    regularizer: Optional[Regularizer] = None


class InputAdiffInequalityConstraints:
    def __init__(self):
        self.initial: list[InputAdiffInequalityConstraint] = []
        self.path: list[InputAdiffInequalityConstraint] = []
        self.terminal: list[InputAdiffInequalityConstraint] = []
        self.control: list[InputAdiffInequalityConstraint] = []


@dataclass
class InputAdiffCost:
    initial: ca.SX = ca.SX(0)
    path: ca.SX = ca.SX(0)
    terminal: ca.SX = ca.SX(0)
