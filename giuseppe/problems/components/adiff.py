from dataclasses import dataclass
from casadi.casadi import Function


@dataclass
class AdiffBoundaryConditions:
    initial: Function
    terminal: Function


@dataclass
class AdiffCost:
    initial: Function
    path: Function
    terminal: Function
