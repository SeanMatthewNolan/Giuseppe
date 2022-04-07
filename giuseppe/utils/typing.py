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


def check_if_any_exist(*args) -> bool:
    return any(arg is not None for arg in args)


def check_if_all_exist(*args) -> bool:
    return any(arg is not None for arg in args)


def sift_nones_from_dict(dict_: dict) -> dict:
    noneless_dict = dict([item for item in dict_.items() if item[1] is not None])
    return noneless_dict
