import numpy as np
from numba import float64
from sympy import Symbol, Matrix, Expr, Integer

NPArray = np.ndarray

EMPTY_ARRAY = np.empty((0,))
EMPTY_2D_ARRAY = np.empty((0, 0))

SymMatrix = Matrix
SymExpr = Expr
SymInteger = Integer

SYM_NULL = Symbol('_not_defined_')
EMPTY_SYM_MATRIX = SymMatrix([])
SYM_ZERO = SymInteger('0')

NumbaFloat = float64
NumbaArray = float64[:]
NumbaMatrix = float64[:, :]
