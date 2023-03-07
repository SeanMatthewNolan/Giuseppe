from typing import Union
from copy import deepcopy

import numpy as np

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import OCP, Dual
from .gauss_newton import gauss_newton


def match_controls_to_primal(
        prob: Union[OCP, Dual], guess: Solution, quadrature: str = 'trapezoidal',
        rel_tol: float = 1e-4, abs_tol: float = 1e-4, verbose: bool = False
) -> Solution:
    """

    Parameters
    ----------
    prob
    guess
    quadrature
    rel_tol
    abs_tol
    verbose

    Returns
    -------

    """
    _num_controls = prob.num_controls

    _compute_boundary_conditions = prob.compute_boundary_conditions
    _compute_dynamics = prob.compute_dynamics

    guess = deepcopy(guess)
    _t, _x, _p, _k = guess.t, guess.x, guess.p, guess.k

    _num_t = len(_t)
    if _num_t < 2:
        raise RuntimeError('Please provide guess with at least 2 points')
    _h_arr = np.diff(_t)

    if quadrature.lower() == 'trapezoidal':

        def _fitting_function(_u_flat: np.ndarray) -> np.ndarray:
            _u = _u_flat.reshape((_num_controls, _num_t))

            res_bc = _compute_boundary_conditions(_t, _x, _p, _k)

            _x_dot = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _u_i in zip(_t, _x.T, _u.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x - _h_arr * (_x_dot[:, :-1] + _x_dot[:, 1:]) / 2

            return np.concatenate((res_bc, res_dyn.flatten()))

    elif quadrature.lower() == 'midpoint':
        _t_mid = (_t[:-1] + _t[1:]) / 2
        _x_mid = (_x[:, :-1] + _x[:, 1:]) / 2

        def _fitting_function(_u_flat: np.ndarray) -> np.ndarray:
            _u = _u_flat.reshape((_num_controls, _num_t))

            _u_mid = (_u[:, :-1] + _u[:, 1:]) / 2

            res_bc = _compute_boundary_conditions(_t, _x, _p, _k)

            _x_dot = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _u_i in zip(_t, _x_mid.T, _u_mid.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x - _h_arr * _x_dot

            return np.concatenate((res_bc, res_dyn.flatten()))

    elif quadrature.lower() == 'simpson':
        _t_mid = (_t[:-1] + _t[1:]) / 2

        def _fitting_function(_u_flat: np.ndarray) -> np.ndarray:
            _u = _u_flat.reshape((_num_controls, _num_t))

            _u_mid = (_u[:, :-1] + _u[:, 1:]) / 2

            res_bc = _compute_boundary_conditions(_t, _x, _p, _k)

            _x_dot_nodes = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _u_i in zip(_t, _x.T, _u.T)
            ]).T

            _x_dot_diff = np.diff(_x_dot_nodes)
            _x_mid = (_x[:, :-1] + _x[:, 1:]) / 2 - _h_arr / 8 * np.diff(_x_dot_nodes)

            _x_dot_mid = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _u_i in zip(_t_mid, _x_mid.T, _u_mid.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x \
                - _h_arr * ((_x_dot_nodes[:, :-1] + _x_dot_nodes[:, 1:]) / 6 + 2 / 3 * _x_dot_mid)

            return np.concatenate((res_bc, res_dyn.flatten()))

    else:
        raise ValueError(f'Quadrature {quadrature} not valid, must be \"trapezoidal\", \"midpoint\", or \"simpson\"')

    _matched = gauss_newton(
            _fitting_function, guess.u.flatten(),
            rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose
    )

    guess.u = _matched.reshape((_num_controls, _num_t))

    return guess


def match_controls_to_control_law(
        prob: Dual, guess: Solution,
        rel_tol: float = 1e-4, abs_tol: float = 1e-4, verbose: bool = False
) -> Solution:
    """

    Parameters
    ----------
    prob
    guess
    quadrature
    rel_tol
    abs_tol
    verbose

    Returns
    -------

    """
    _num_controls = prob.num_controls

    _compute_control_law = prob.compute_control_law

    guess = deepcopy(guess)
    _t, _x, _lam, _p, _k = guess.t, guess.x, guess.lam, guess.p, guess.k

    _num_t = len(_t)
    if _num_t < 2:
        raise RuntimeError('Please provide guess with at least 2 points')
    _h_arr = np.diff(_t)

    def _fitting_function(_u_flat: np.ndarray) -> np.ndarray:
        _u = _u_flat.reshape((_num_controls, _num_t))

        _dh_du = np.array([
            _compute_control_law(_t_i, _x_i, _lam_i, _u_i, _p, _k)
            for _t_i, _x_i, _lam_i, _u_i in zip(_t, _x.T, _lam.T, _u.T)
        ]).T

        return _dh_du.flatten()

    _matched = gauss_newton(_fitting_function, guess.u.flatten(), rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)
    guess.u = _matched.reshape((_num_controls, _num_t))

    return guess

