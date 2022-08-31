from dataclasses import dataclass
from typing import Optional

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


@dataclass
class InputAdiffConstraints:
    initial: ca.SX = ca.SX.sym('', 0)
    terminal: ca.SX = ca.SX.sym('', 0)


@dataclass
class InputAdiffInequalityConstraint:
    expr: ca.SX
    lower_limit: ca.SX
    upper_limit: ca.SX
    regularizer: Optional[Regularizer] = None


class InputAdiffInequalityConstraints:
    def __init__(self):
        self.initial: list[InputAdiffInequalityConstraints] = []
        self.path: list[InputAdiffInequalityConstraints] = []
        self.terminal: list[InputAdiffInequalityConstraints] = []
        self.control: list[InputAdiffInequalityConstraints] = []


@dataclass
class InputAdiffCost:
    initial: ca.SX = ca.SX(0)
    path: ca.SX = ca.SX(0)
    terminal: ca.SX = ca.SX(0)
