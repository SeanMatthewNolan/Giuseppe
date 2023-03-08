from typing import Union
from copy import deepcopy

import numpy as np

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual, Adjoints
from giuseppe.utils import make_array_slices
from .gauss_newton import gauss_newton


def match_constants_to_boundary_conditions(
        prob: Union[BVP, OCP, Dual], guess: Solution, rel_tol: float = 1e-4, abs_tol: float = 1e-4,
        use_adjoint_bcs: bool = False, verbose: bool = False
) -> Solution:
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
    use_adjoint_bcs : bool, default=False
    verbose : bool, default=False

    Returns
    -------
    guess with projected constants

    """
    guess = deepcopy(guess)

    _compute_initial_boundary_conditions = prob.compute_initial_boundary_conditions
    _compute_terminal_boundary_conditions = prob.compute_terminal_boundary_conditions
    _t, _x, _p = guess.t, guess.x, guess.p

    if use_adjoint_bcs and isinstance(prob, Dual):
        _compute_initial_adjoint_boundary_conditions = prob.compute_initial_adjoint_boundary_conditions
        _compute_terminal_adjoint_boundary_conditions = prob.compute_terminal_adjoint_boundary_conditions
        _u, _lam, _nu0, _nuf = guess.u, guess.lam, guess.nu0, guess.nuf

        def _constraint_function(_k: np.ndarray):
            return np.concatenate((
                _compute_initial_boundary_conditions(_t[0], _x[:, 0], _p, _k),
                _compute_terminal_boundary_conditions(_t[-1], _x[:, -1], _p, _k),
                _compute_initial_adjoint_boundary_conditions(_t[0], _x[:, 0], _lam[:, 0], _u[:, 0], _p, _nu0, _k),
                _compute_terminal_adjoint_boundary_conditions(_t[-1], _x[:, -1], _lam[:, -1], _u[:, -1], _p, _nuf, _k),
            ))
    else:
        def _constraint_function(_k: np.ndarray):
            return np.concatenate((
                _compute_initial_boundary_conditions(_t[0], _x[:, 0], _p, _k),
                _compute_terminal_boundary_conditions(_t[-1], _x[:, -1], _p, _k),
            ))

    guess.k = gauss_newton(_constraint_function, guess.k,
                           rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)
    return guess


def match_states_to_boundary_conditions(
        prob: Union[BVP, OCP, Dual], guess: Solution, rel_tol: float = 1e-4, abs_tol: float = 1e-4,
        verbose: bool = False
) -> Solution:
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
    verbose : bool, default=False

    Returns
    -------
    guess with projected independent, states, and parameters

    """
    guess = deepcopy(guess)

    _initial_node_spacing = (guess.t - guess.t[0]) / (guess.t[-1] - guess.t[0])

    _compute_initial_boundary_conditions = prob.compute_initial_boundary_conditions
    _compute_terminal_boundary_conditions = prob.compute_terminal_boundary_conditions
    _num_boundaries, _num_states, _num_parameters = 2, prob.num_states, prob.num_parameters
    _k = guess.k

    _t_slice, _x_slice, _p_slice = make_array_slices((_num_boundaries, _num_states * _num_boundaries, _num_parameters))

    def _constraint_function(_z: np.ndarray):
        _t = _z[_t_slice]
        _x = _z[_x_slice].reshape((_num_states, _num_boundaries))
        _p = _z[_p_slice]

        return np.concatenate((
                _compute_initial_boundary_conditions(_t[0], _x[:, 0], _p, _k),
                _compute_terminal_boundary_conditions(_t[-1], _x[:, -1], _p, _k),
        ))

    # Converting supplied guess to initial vector for Gauss-Newton method
    _idx_t = tuple(np.linspace(0, guess.t.shape[0] - 1, _num_boundaries, dtype=int))
    _idx_x = tuple(np.linspace(0, guess.x.shape[1] - 1, _num_boundaries, dtype=int))
    _slp_guess = []
    for _idx in _idx_t:
        _slp_guess.append([guess.t[_idx]])
    for _idx in _idx_x:
        _slp_guess.append(guess.x[:, _idx])
    _slp_guess.append(guess.p)
    _slp_guess = np.concatenate(_slp_guess)

    # Application of Gauss-Newton
    out = gauss_newton(_constraint_function, _slp_guess, rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)

    # Assigning fitted values to guess to output
    matched_t = out[_t_slice]
    matched_x = out[_x_slice].reshape((_num_states, _num_boundaries))

    guess.t = matched_t[0] + (matched_t[-1] - matched_t[0]) * _initial_node_spacing
    guess.x = (matched_x[:, 0] + np.outer(_initial_node_spacing, (matched_x[:, -1] - matched_x[:, 0]))).T
    guess.p = out[_p_slice]

    return guess


def match_adjoints_to_boundary_conditions(
        prob: Union[Adjoints, Dual], guess: Solution, rel_tol: float = 1e-4, abs_tol: float = 1e-4,
        verbose: bool = False
) -> Solution:
    """
    Projects the constant array of a guess to the problem's boundary conditions to get the closest match

    Parameters
    ----------
    prob : Dual, Adjoints
        problem whose BCs are to be matched
    guess : Solution
        guess from which to match the independent, states, and parameters
    abs_tol : float, default=1e-4
       absolute tolerance
    rel_tol : float, default=1e-4
       relative tolerance
    verbose : bool, default=False

    Returns
    -------
    guess with projected independent, states, and parameters

    """
    guess = deepcopy(guess)

    _initial_node_spacing = (guess.t - guess.t[0]) / (guess.t[-1] - guess.t[0])

    _num_costates = prob.num_costates
    _num_initial_adjoints = prob.num_initial_adjoints
    _num_terminal_adjoints = prob.num_terminal_adjoints

    _compute_initial_adjoint_boundary_conditions = prob.compute_initial_adjoint_boundary_conditions
    _compute_terminal_adjoint_boundary_conditions = prob.compute_terminal_adjoint_boundary_conditions
    _t, _x, _u, _p, _k = guess.t, guess.x, guess.u, guess.p, guess.k

    _lam_slice, _nu0_slice, _nuf_slice = make_array_slices(
            (2 * _num_costates, _num_initial_adjoints, _num_terminal_adjoints))

    def _constraint_function(_adjoints: np.ndarray):
        _lam = _adjoints[_lam_slice].reshape((_num_costates, 2))
        _nu0 = _adjoints[_nu0_slice]
        _nuf = _adjoints[_nuf_slice]

        return np.concatenate((
            _compute_initial_adjoint_boundary_conditions(_t[0], _x[:, 0], _lam[:, 0], _u[:, 0], _p, _nu0, _k),
            _compute_terminal_adjoint_boundary_conditions(_t[-1], _x[:, -1], _lam[:, -1], _u[:, -1], _p, _nuf, _k),
        ))

    # Converting supplied guess to initial vector for Gauss-Newton method
    _slp_guess = np.concatenate((guess.lam[:, 0], guess.lam[:, -1], guess.nu0, guess.nuf))

    # Application of Gauss-Newton
    out = gauss_newton(
            _constraint_function, _slp_guess, rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)

    # Assigning fitted values to guess to output
    matched_lam = out[_lam_slice].reshape((_num_costates, 2))

    guess.lam = (matched_lam[:, 0] + np.outer(_initial_node_spacing, (matched_lam[:, -1] - matched_lam[:, 0]))).T
    guess.nu0 = out[_nu0_slice]
    guess.nuf = out[_nuf_slice]

    return guess
