from .aliases import SymMatrix, SymExpr


def as_scalar(single_element_matrix: SymMatrix) -> SymExpr:
    if isinstance(single_element_matrix, SymMatrix) and (len(single_element_matrix) == 1):
        return single_element_matrix[0, 0]
    else:
        raise ValueError(f'{single_element_matrix} not a single element symbolic matrix')
