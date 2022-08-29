from dataclasses import dataclass
from typing import Optional

import casadi as ca

from giuseppe.problems.regularization import Regularizer


@dataclass
class InputConstant:
    var: ca.SX
    default_value: float = 0.


@dataclass
class InputNamedExpr:
    var: ca.SX
    expr: ca.SX


class InputConstraints:
    def __init__(self):
        self.initial: ca.SX = ca.SX.sym('', 0)
        self.terminal: ca.SX = ca.SX.sym('', 0)


@dataclass
class InputInequalityConstraint:
    expr: ca.SX
    lower_limit: ca.SX
    upper_limit: ca.SX
    regularizer: Optional[Regularizer] = None


class InputInequalityConstraints:
    def __init__(self):
        self.initial: list[InputInequalityConstraint] = []
        self.path: list[InputInequalityConstraint] = []
        self.terminal: list[InputInequalityConstraint] = []
        self.control: list[InputInequalityConstraint] = []


@dataclass
class InputCost:
    initial: ca.SX = ca.SX(0)
    path: ca.SX = ca.SX(0)
    terminal: ca.SX = ca.SX(0)
