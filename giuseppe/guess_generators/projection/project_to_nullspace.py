from copy import copy
from typing import Union, Callable, Optional

import numpy as np

from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, BVPSol, OCPSol, DualOCPSol
from giuseppe.utils.numerical_derivatives.finite_difference import central_difference_jacobian as jac, ArrayFunction


SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP]
SUPPORTED_SOLUTIONS = Union[BVPSol, OCPSol, DualOCPSol]


# TODO Use Armijo step etc. to make more stable
# TODO Explore options based on stability (linear vs. nonlinear)
def project_to_nullspace(func: ArrayFunction, arr: np.ndarray, max_steps: Optional[int] = 8,
                         abs_tol: float = 1e-3, rel_tol: float = 1e-3) -> np.ndarray:
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

    Returns
    -------
    projected array

    """

    arr = np.array(arr)

    converged, step_num = False, 1
    while not converged:
        arr_last = copy(arr)

        residual = func(arr)
        sensitivity = jac(func, arr)
        p_inv_sensitivity = np.linalg.pinv(sensitivity)

        arr -= p_inv_sensitivity @ residual

        if np.linalg.norm(arr - arr_last) < (abs_tol + rel_tol * np.linalg.norm(arr)):
            converged = True

        else:
            step_num += 1
            if step_num > max_steps:
                raise RuntimeError('Projection did not converge!')

    return arr
