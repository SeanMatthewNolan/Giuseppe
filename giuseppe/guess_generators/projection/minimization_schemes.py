from copy import copy
from typing import Union, Optional

import numpy as np

from giuseppe.problems import CompBVP, CompOCP, CompDualOCP
from giuseppe.utils.numerical_derivatives.finite_difference import central_difference_jacobian as jac, ArrayFunction

SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP]


# TODO Use Armijo step etc. to make more stable
# TODO Explore options based on stability (linear vs. nonlinear)
def project_to_nullspace(func: ArrayFunction, arr: np.ndarray, max_steps: Optional[int] = 8,
                         abs_tol: float = 1e-3, rel_tol: float = 1e-3, backtrack: bool = True) -> np.ndarray:
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
    backtrack : bool, default=True
        Whether to use backtracking line search during minimization

    Returns
    -------
    projected array

    """

    # Backtracking constants (Same as SciPy's SolveBVP)
    min_improvement = 0.2  # Min. relative improvement (Armijo constant)
    backtrack_decrease_factor = 0.5  # Step size decrease factor
    max_backtrack_steps = 4

    arr = np.array(arr)
    residual = func(arr)
    cost = np.dot(residual, residual)

    converged, step_num = False, 1
    while not converged:
        arr_last = copy(arr)
        cost_last = copy(cost)

        sensitivity = jac(func, arr)
        p_inv_sensitivity = np.linalg.pinv(sensitivity)

        del_arr = -p_inv_sensitivity @ residual
        step_size = 1

        for step in range(max_backtrack_steps):
            arr = arr_last + step_size * del_arr
            residual = func(arr)
            cost = np.dot(residual, residual)

            if cost < (1 - 2 * min_improvement * step_size) * cost_last or not backtrack:
                break

            step_size = backtrack_decrease_factor * step_size

        arr -= p_inv_sensitivity @ residual

        if np.linalg.norm(arr - arr_last) < (abs_tol + rel_tol * np.linalg.norm(arr)):
            converged = True

        else:
            step_num += 1
            if step_num > max_steps:
                raise RuntimeError('Projection did not converge!')

    return arr


def gradient_descent(func: ArrayFunction, arr: np.ndarray, max_steps: Optional[int] = 8,
                     abs_tol: float = 1e-3, backtrack: bool = True) -> np.ndarray:
    """
    Function which minimizes the squared norm of the residuals via gradient descent method

    Parameters
    ----------
    func : ArrayFunction
        residual function (desire to minimize the norm squared of this function)
    arr : np.ndarray
        initial guess & default values of arguments to minimize the objective function
    max_steps : int, default=8
        maximum number of steps the iterative solver will take
    abs_tol : float, default=1e-3
        absolute tolerance
    backtrack : bool, default=True
        Whether to use backtracking line search during minimization

    Returns
    -------
    arr : argmin of objective function

    """

    def obj_func(_arr):
        return np.dot(func(_arr), func(_arr))

    def grad_func(_arr):
        return 2 * jac(func, _arr).T @ func(_arr)

    arr = np.array(arr)
    obj = obj_func(arr)
    grad = grad_func(arr)

    # Backtracking constants (Same as SciPy's SolveBVP)
    min_improvement = 0.2  # Min. relative improvement (Armijo constant)
    backtrack_decrease_factor = 0.25  # Step size decrease factor
    max_backtrack_steps = 4

    converged, step_num = False, 1
    while not converged:
        arr_last = copy(arr)
        obj_last = copy(obj)
        grad_last = copy(grad)

        del_arr = -grad_last
        step_size = 1

        for step in range(max_backtrack_steps):
            arr = arr_last + step_size * del_arr
            obj = obj_func(arr)

            if obj < obj_last + min_improvement * step_size * grad_last.T @ del_arr or not backtrack:
                break

            step_size = backtrack_decrease_factor * step_size

        if abs(obj - obj_last) < abs_tol:
            converged = True

        else:
            step_num += 1
            if step_num > max_steps:
                raise RuntimeError('Projection did not converge!')

    return arr


def newtons_method(func: ArrayFunction, arr: np.ndarray, max_steps: Optional[int] = 8,
                   abs_tol: float = 1e-3, backtrack: bool = True) -> np.ndarray:
    """
    Function which minimizes the squared norm of the residuals via gradient descent method

    Parameters
    ----------
    func : ArrayFunction
        residual function (desire to minimize the norm squared of this function)
    arr : np.ndarray
        initial guess & default values of arguments to minimize the objective function
    max_steps : int, default=8
        maximum number of steps the iterative solver will take
    abs_tol : float, default=1e-3
        absolute tolerance
    backtrack : bool, default=True
        Whether to use backtracking line search during minimization

    Returns
    -------
    arr : argmin of objective function

    """

    def obj_func(_arr):
        return np.dot(func(_arr), func(_arr))

    def grad_func(_arr):
        return 2 * jac(func, _arr).T @ func(_arr)

    def hess_func(_arr):
        return jac(grad_func, _arr)

    arr = np.array(arr)
    obj = obj_func(arr)
    grad = grad_func(arr)
    inv_hess = np.linalg.pinv(hess_func(arr))

    min_improvement = 0.2  # Min. relative improvement (Armijo constant)
    backtrack_decrease_factor = 0.25  # Step size decrease factor
    max_backtrack_steps = 4

    step_num = 1
    while True:
        arr_last = copy(arr)
        obj_last = copy(obj)
        grad_last = copy(grad)
        inv_hess_last = copy(inv_hess)

        del_arr = -inv_hess_last @ grad_last
        lam2 = np.dot(grad_last, -del_arr)

        if lam2 < abs_tol:
            break

        step_size = 1

        for step in range(max_backtrack_steps):
            arr = arr_last + step_size * del_arr
            obj = obj_func(arr)

            if obj < obj_last - min_improvement * step_size * lam2 or not backtrack:
                break

            step_size = backtrack_decrease_factor * step_size

        step_num += 1
        if step_num > max_steps:
            raise RuntimeError('Projection did not converge!')

    return arr
