from copy import copy
from typing import Union, Optional, Callable

import numpy as np

from giuseppe.utils.numerical_derivatives.finite_difference\
    import forward_difference_jacobian, central_difference_jacobian, backward_difference_jacobian, ArrayFunction


# TODO Use Armijo step etc. to make more stable
# TODO Explore options based on stability (linear vs. nonlinear)
def sequential_linearized_projection(
        func: ArrayFunction, arr: np.ndarray, max_steps: Optional[int] = 8,
        abs_tol: float = 1e-4, rel_tol: float = 1e-4,
        jacobian_function: Union[str, Callable] = 'central'
) -> np.ndarray:
    """
    Function which projects an array onto the nullspace of a function

    Parameters
    ----------
    func : ArrayFunction
        function onto whose nullspace to project, i.e. arr will be set to bring output closest to zero
    arr : np.ndarray
        array which to project onto nullspace
        both serves as the initial guess and the array of default values
    max_steps : int, default=8
        maximum number of steps the iterative solver will take
    abs_tol : float, default=1e-3
        absolute tolerance
    rel_tol : float, default=1e-3
        relative tolerance
    jacobian_function : str, Callable, default='central'
        String inputs of 'central', 'forward', or 'backward' specify corresponding numerical derivatives to be taken
        Providing a Callable will use that function to compute the Jacobian

    Returns
    -------
    projected array

    """
    
    if jacobian_function == 'central':
        compute_jacobian = central_difference_jacobian
    elif jacobian_function == 'forward':
        compute_jacobian = forward_difference_jacobian
    elif jacobian_function == 'backward':
        compute_jacobian = backward_difference_jacobian
    elif isinstance(jacobian_function, Callable):
        compute_jacobian = jacobian_function
    else:
        raise ValueError('\'jacobian_function\' should be defined as central, forward, or backward or be a Callable '
                         'which takes the current array and return the Jacobian')

    arr = np.array(arr)
    rel_tol_threshold = rel_tol * np.linalg.norm(arr)

    converged, step_num = False, 1
    while not converged:
        arr_last = copy(arr)

        residual = func(arr)
        sensitivity = compute_jacobian(func, arr)
        p_inv_sensitivity = np.linalg.pinv(sensitivity)

        arr -= p_inv_sensitivity @ residual

        # TODO Check about enforcing this constant
        if np.linalg.norm(arr - arr_last) < (abs_tol + rel_tol_threshold):
            converged = True

        else:
            step_num += 1
            if step_num > max_steps:
                raise RuntimeError('Projection did not converge!')

    return arr
