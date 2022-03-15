from numba import float64
from numpy import ndarray
from sympy import Symbol, Matrix, Expr, Integer

Array = ndarray

SymMatrix = Matrix
SymExpr = Expr
SymInteger = Integer

SYM_NULL = Symbol('_not_defined_')
EMPTY_SYM_MATRIX = SymMatrix([])
SYM_ZERO = SymInteger('0')

NumbaFloat = float64
NumbaArray = float64[:]
NumbaMatrix = float64[:, :]

