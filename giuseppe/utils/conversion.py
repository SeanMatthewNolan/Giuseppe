import numpy as np

from .typing import SymMatrix, SymExpr


def matrix_as_vector(matrix_1d: SymMatrix) -> np.ndarray:
    if isinstance(matrix_1d, SymMatrix) and (1 in matrix_1d.shape):
        return np.array(matrix_1d.flat())
    else:
        raise TypeError(f'{matrix_1d} is not a single column/row symbolic matrix so can\'t collapse to array vector')


def matrix_as_scalar(single_element_matrix: SymMatrix) -> SymExpr:
    if isinstance(single_element_matrix, SymMatrix) and (len(single_element_matrix) == 1):
        return single_element_matrix[0, 0]
    else:
        raise TypeError(f'{single_element_matrix} not a single element symbolic matrix')
