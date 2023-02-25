from copy import copy
from typing import Union, Optional, Callable
from warnings import warn

import numpy as np

from giuseppe.utils.numerical_derivatives.finite_difference\
    import forward_difference_jacobian, central_difference_jacobian, backward_difference_jacobian, ArrayFunction


# TODO Use Armijo step etc. to make more stable
# TODO Explore options based on stability (linear vs. nonlinear)
def sequential_linearized_projection(
        func: ArrayFunction, arr: np.ndarray, max_steps: int = 20,
        abs_tol: float = 1e-4, rel_tol: float = 1e-4,
        jacobian_function: Union[str, Callable] = 'central',
        use_line_search: bool = True, line_search_alpha: float = 1e-4, line_search_reduction_ratio: float =  0.5,
        verbose: bool = True,
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
    abs_tol : float, default=1e-4
        absolute tolerance
    rel_tol : float, default=1e-4
        relative tolerance
    jacobian_function : str, Callable, default='central'
        String inputs of 'central', 'forward', or 'backward' specify corresponding numerical derivatives to be taken
        Providing a Callable will use that function to compute the Jacobian
    use_line_search : bool, default=True
        whether to use Armijo backtracking line search
    line_search_alpha : float, default=1e-4
        "alpha" variable for line search (defines sufficient descent)
    line_search_reduction_ratio : float, default=0.5
        ratio to decrease step in each iteration of line search
    verbose :  bool, default=False
        whether to print current status

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

    arr = np.asarray(arr)
    step_tol = abs(abs_tol + rel_tol * arr)
    residual = func(arr)
    residual_tol = abs(abs_tol + rel_tol * residual)
    if np.all(abs(residual) < residual_tol):
        if verbose:
            print(f'Residual tolerance already satisfied, forgoing SLP')
        return arr

    if verbose:
        print(f'Starting SLP: InitRes = {np.linalg.norm(residual)}, ArraySize = {len(arr)}')

    for step_num in range(1, max_steps + 1):
        sensitivity = compute_jacobian(func, arr)
        p_inv_sensitivity = np.linalg.pinv(sensitivity)

        raw_step = p_inv_sensitivity @ residual

        if np.all(abs(raw_step) < step_tol):
            if verbose:
                print(f'SLP converged in {step_num-1} steps with final step size {np.linalg.norm(raw_step)}\n')
            return arr

        if use_line_search:
            arr, residual = armijo_step(arr, residual, raw_step, func, verbose=verbose,
                                        alpha=line_search_alpha, step_reduction_ratio=line_search_reduction_ratio)
        else:
            arr -= raw_step
            residual = func(arr)

        if verbose:
            print(f'    Iter {step_num}: ResNorm = {np.linalg.norm(residual)}; StepSize = {np.linalg.norm(raw_step)}')

        if np.all(abs(residual) < residual_tol):
            if verbose:
                print(f'SLP converged in {step_num} steps with final residual {np.linalg.norm(residual)}\n')
            return arr

    print(f'Projection failed to converge in {max_steps} steps! ❌\n')
    return arr


def armijo_step(
        current_arr: np.ndarray, current_residual: np.ndarray, raw_step: np.ndarray, func: Callable,
        alpha: float = 1e-4, step_reduction_ratio: float = 0.5, min_step_ratio: float = 0.000001, verbose: bool = False
) -> (np.ndarray, np.ndarray):

    step, step_ratio = raw_step, 1
    norm_current_residual = np.linalg.norm(current_residual)
    trial_arr = current_arr - raw_step
    trial_residual = func(trial_arr)

    while np.linalg.norm(trial_residual) > (1 - alpha * step_ratio) * norm_current_residual:
        step_ratio *= step_reduction_ratio
        trial_arr -= step_ratio * raw_step
        trial_residual = func(trial_arr)

        if step_ratio < min_step_ratio:
            break

    if verbose:
        print(f'        Armijo step ratio = {step_ratio}')

    return trial_arr, trial_residual
