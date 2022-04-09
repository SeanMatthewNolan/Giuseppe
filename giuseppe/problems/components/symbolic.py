from dataclasses import dataclass

from giuseppe.utils.typing import SymMatrix, EMPTY_SYM_MATRIX, SymExpr, SYM_ZERO


@dataclass
class SymBoundaryConditions:
    initial: SymMatrix = EMPTY_SYM_MATRIX
    terminal: SymMatrix = EMPTY_SYM_MATRIX


@dataclass
class SymCost:
    initial: SymExpr = SYM_ZERO
    path: SymExpr = SYM_ZERO
    terminal: SymExpr = SYM_ZERO
