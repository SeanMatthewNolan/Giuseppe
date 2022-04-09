from dataclasses import dataclass
from typing import Optional

from giuseppe.problems.regularization import Regularizer


@dataclass
class InputState:
    name: str
    eom: str


@dataclass
class InputConstant:
    name: str
    default_value: float = 0.


class InputConstraints:
    def __init__(self):
        self.initial: list[str] = []
        self.terminal: list[str] = []


@dataclass
class InputInequalityConstraint:
    expr: str
    lower_limit: str
    upper_limit: str
    regularizer: Optional[Regularizer] = None


class InputInequalityConstraints:
    def __init__(self):
        self.initial: list[InputInequalityConstraint] = []
        self.path: list[InputInequalityConstraint] = []
        self.terminal: list[InputInequalityConstraint] = []
        self.control: list[InputInequalityConstraint] = []


@dataclass
class InputCost:
    initial: str = '0'
    path: str = '0'
    terminal: str = '0'