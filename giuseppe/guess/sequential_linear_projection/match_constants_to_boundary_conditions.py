from typing import Union
from copy import deepcopy

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual
from .sequential_linear_projection import sequential_linearized_projection


def match_constants_to_boundary_conditions(
        prob: Union[BVP, OCP, Dual], guess: Solution, rel_tol: float = 1e-4, abs_tol: float = 1e-4) -> Solution:
    """
    Projects the constant array of a guess to the problem's boundary conditions to get the closest match

    Parameters
    ----------
    prob : BVP or OCP or Dual
        problem whose BCs are to be matched
    guess : Solution
        guess from which to match the constant
    abs_tol : float, default=1e-4
       absolute tolerance
    rel_tol : float, default=1e-4
       relative tolerance

    Returns
    -------
    guess with projected constants

    """
    _compute_boundary_conditions = prob.compute_boundary_conditions
    _t, _x, _p = guess.t, guess.x, guess.p

    def _constraint_function(_k):
        return _compute_boundary_conditions(_t, _x, _p, _k)

    guess = deepcopy(guess)
    guess.k = sequential_linearized_projection(_constraint_function, guess.k, rel_tol=rel_tol, abs_tol=abs_tol)
    return guess
