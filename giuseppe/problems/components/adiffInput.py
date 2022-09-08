from dataclasses import dataclass
from typing import Optional, Union

import casadi as ca
import numpy as np

from giuseppe.problems.regularization import Regularizer


@dataclass
class InputAdiffState:
    states: ca.MX = ca.MX.sym('', 0)
    eoms: ca.MX = ca.MX.sym('', 0)


@dataclass
class InputAdiffConstant:
    constants: ca.MX = ca.MX.sym('', 0)
    default_values: np.ndarray = np.empty((0, 1))


@dataclass
class InputAdiffConstraints:
    initial: ca.MX = ca.MX.sym('', 0)
    terminal: ca.MX = ca.MX.sym('', 0)


@dataclass
class InputAdiffInequalityConstraint:
    expr: ca.MX
    lower_limit: Union[ca.MX, float]
    upper_limit: Union[ca.MX, float]
    regularizer: Optional[Regularizer] = None


class InputAdiffInequalityConstraints:
    def __init__(self):
        self.initial: list[InputAdiffInequalityConstraint] = []
        self.path: list[InputAdiffInequalityConstraint] = []
        self.terminal: list[InputAdiffInequalityConstraint] = []
        self.control: list[InputAdiffInequalityConstraint] = []


@dataclass
class InputAdiffCost:
    initial: ca.MX = ca.MX(0)
    path: ca.MX = ca.MX(0)
    terminal: ca.MX = ca.MX(0)
