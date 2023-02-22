from typing import Union
from copy import deepcopy

import numpy as np

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
    guess = deepcopy(guess)

    _compute_boundary_conditions = prob.compute_boundary_conditions
    _t, _x, _p = guess.t, guess.x, guess.p

    def _constraint_function(_k: np.ndarray):
        return _compute_boundary_conditions(_t, _x, _p, _k)

    guess.k = sequential_linearized_projection(_constraint_function, guess.k, rel_tol=rel_tol, abs_tol=abs_tol)
    return guess


def match_states_to_boundary_conditions(
        prob: Union[BVP, OCP, Dual], guess: Solution, rel_tol: float = 1e-4, abs_tol: float = 1e-4) -> Solution:
    """
    Projects the constant array of a guess to the problem's boundary conditions to get the closest match

    Parameters
    ----------
    prob : BVP or OCP or Dual
        problem whose BCs are to be matched
    guess : Solution
        guess from which to match the independent, states, and parameters
    abs_tol : float, default=1e-4
       absolute tolerance
    rel_tol : float, default=1e-4
       relative tolerance

    Returns
    -------
    guess with projected independent, states, and parameters

    """
    guess = deepcopy(guess)

    _compute_boundary_conditions = prob.compute_boundary_conditions
    _num_boundaries, _num_states, _num_parameters = prob.num_arcs + 1, prob.num_states, prob.num_parameters
    _k = guess.k

    _t_slice = slice(_num_boundaries)
    _x_slice = slice(_t_slice.stop, _t_slice.stop + _num_states * _num_boundaries)
    _p_slice = slice(_x_slice.stop, _x_slice.stop + _num_parameters)

    def _constraint_function(_z: np.ndarray):
        _t = _z[_t_slice]
        _x = _z[_x_slice].reshape((_num_states, _num_boundaries))
        _p = _z[_p_slice]

        return _compute_boundary_conditions(_t, _x, _p, _k)

    # Converting supplied guess to initial vector for SLP method
    _idx_t = tuple(np.linspace(0, guess.t.shape[0] - 1, _num_boundaries, dtype=int))
    _idx_x = tuple(np.linspace(0, guess.x.shape[1] - 1, _num_boundaries, dtype=int))
    _slp_guess = []
    for _idx in _idx_t:
        _slp_guess.append([guess.t[_idx]])
    for _idx in _idx_x:
        _slp_guess.append(guess.x[:, _idx])
    # if guess.p.size > 0:
    _slp_guess.append(guess.p)
    _slp_guess = np.concatenate(_slp_guess)

    # Application of SLP
    out = sequential_linearized_projection(_constraint_function, _slp_guess, rel_tol=rel_tol, abs_tol=abs_tol)

    # Assigning fitted values to guess to output
    guess.t = out[_t_slice]
    guess.x = out[_x_slice].reshape((_num_states, _num_boundaries))
    guess.p = out[_p_slice]

    return guess
