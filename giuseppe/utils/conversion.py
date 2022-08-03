import numpy as np

from giuseppe.utils.typing import SymMatrix, SymExpr


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


def arrays_to_lists_in_dict(dict_obj: dict):
    for key, val in dict_obj.items():
        if isinstance(val, np.ndarray):
            dict_obj[key] = val.tolist()
        elif isinstance(val, dict):
            dict_obj[key] = arrays_to_lists_in_dict(val)
    return dict_obj


def lists_to_arrays_in_dict(dict_obj: dict):
    for key, val in dict_obj.items():
        if isinstance(val, list):
            dict_obj[key] = np.asarray(val)
        elif isinstance(val, dict):
            dict_obj[key] = lists_to_arrays_in_dict(val)
    return dict_obj
