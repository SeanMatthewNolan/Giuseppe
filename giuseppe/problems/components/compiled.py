from dataclasses import dataclass
from typing import Callable


@dataclass
class CompBoundaryConditions:
    initial: Callable
    terminal: Callable


@dataclass
class CompCost:
    initial: Callable
    path: Callable
    terminal: Callable