from typing import Union
from copy import deepcopy

import numpy as np

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual
from giuseppe.utils import make_array_slices
from .gauss_newton import gauss_newton


def match_primal(
        prob: Union[BVP, OCP, Dual], guess: Solution, quadrature: str = 'trapezoidal',
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
    _num_states = prob.num_states
    _num_parameters = prob.num_parameters

    _compute_initial_boundary_conditions = prob.compute_initial_boundary_conditions
    _compute_terminal_boundary_conditions = prob.compute_terminal_boundary_conditions

    if prob.prob_class == 'bvp':
        def _compute_dynamics(_t_i, _x_i, _, _p, _k):
            return prob.compute_dynamics(_t_i, _x_i, _p, _k)
    else:
        _compute_dynamics = prob.compute_dynamics

    guess = deepcopy(guess)
    _input_t, _u, _k = guess.t, guess.u, guess.k

    _num_t = len(_input_t)
    if _num_t < 2:
        raise RuntimeError('Please provide guess with at least 2 points')

    _tau = (_input_t - _input_t[0]) / (_input_t[-1] - _input_t[0])
    _h_arr = np.diff(_tau)

    _, _x_slice, _p_slice = make_array_slices((2, _num_t * _num_states, _num_parameters))

    if quadrature.lower() == 'trapezoidal':

        def _fitting_function(_z: np.ndarray) -> np.ndarray:
            _t0, _tf = _z[0], _z[-1]
            _x = _z[_x_slice].reshape((_num_states, _num_t))
            _p = _z[_p_slice]

            _delta_t = (_tf - _t0)
            _t = _delta_t * _tau + _t0

            res_bc = np.concatenate((
                _compute_initial_boundary_conditions(_t0, _x[:, 0], _p, _k),
                _compute_terminal_boundary_conditions(_tf, _x[:, -1], _p, _k),
            ))

            _x_dot = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _u_i in zip(_t, _x.T, _u.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x - _h_arr * (_x_dot[:, :-1] + _x_dot[:, 1:]) / 2

            return np.concatenate((res_bc, res_dyn.flatten()))

    elif quadrature.lower() == 'midpoint':
        _tau_mid = (_tau[:-1] + _tau[1:]) / 2
        _u_mid = (_u[:, :-1] + _u[:, 1:]) / 2

        def _fitting_function(_z: np.ndarray) -> np.ndarray:
            _t0, _tf = _z[0], _z[-1]
            _x = _z[_x_slice].reshape((_num_states, _num_t))
            _p = _z[_p_slice]

            _delta_t = (_tf - _t0)
            _t = _delta_t * _tau + _t0
            _t_mid = _delta_t * _tau_mid + _t0

            _x_mid = (_x[:, :-1] + _x[:, 1:]) / 2

            res_bc = np.concatenate((
                _compute_initial_boundary_conditions(_t0, _x[:, 0], _p, _k),
                _compute_terminal_boundary_conditions(_tf, _x[:, -1], _p, _k),
            ))

            _x_dot = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _u_i in zip(_input_t, _x_mid.T, _u_mid.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x - _h_arr * _x_dot

            return np.concatenate((res_bc, res_dyn.flatten()))

    elif quadrature.lower() == 'simpson':
        _compute_dynamics = prob.compute_dynamics

        _tau_mid = (_tau[:-1] + _tau[1:]) / 2
        _u_mid = (_u[:, :-1] + _u[:, 1:]) / 2

        def _fitting_function(_z: np.ndarray) -> np.ndarray:
            _t0, _tf = _z[0], _z[-1]
            _x = _z[_x_slice].reshape((_num_states, _num_t))
            _p = _z[_p_slice]

            _delta_t = (_tf - _t0)
            _t = _delta_t * _tau + _t0
            _t_mid = _delta_t * _tau_mid + _t0

            res_bc = np.concatenate((
                _compute_initial_boundary_conditions(_t0, _x[:, 0], _p, _k),
                _compute_terminal_boundary_conditions(_tf, _x[:, -1], _p, _k),
            ))

            _x_dot_nodes = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _u_i in zip(_input_t, _x.T, _u.T)
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
            _fitting_function, np.concatenate(((guess.t[0], guess.t[-1]), guess.x.flatten(), guess.p)),
            rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose
    )

    guess.t = _tau * (_matched[-1] - _matched[0]) + _matched[0]
    guess.x = _matched[_x_slice].reshape((_num_states, _num_t))
    guess.p = _matched[_p_slice]

    return guess


def match_primal_and_constant_control(
        prob: Union[BVP, OCP, Dual], guess: Solution, quadrature: str = 'trapezoidal',
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
    _num_states = prob.num_states
    _num_controls = prob.num_controls
    _num_parameters = prob.num_parameters

    _compute_initial_boundary_conditions = prob.compute_initial_boundary_conditions
    _compute_terminal_boundary_conditions = prob.compute_terminal_boundary_conditions

    if prob.prob_class == 'bvp':
        def _compute_dynamics(_t_i, _x_i, _, _p, _k):
            return prob.compute_dynamics(_t_i, _x_i, _p, _k)
    else:
        _compute_dynamics = prob.compute_dynamics

    guess = deepcopy(guess)
    _input_t, _k = guess.t, guess.k

    _num_t = len(_input_t)
    if _num_t < 2:
        raise RuntimeError('Please provide guess with at least 2 points')

    _tau = (_input_t - _input_t[0]) / (_input_t[-1] - _input_t[0])
    _h_arr = np.diff(_tau)

    _, _x_slice, _u_slice, _p_slice = make_array_slices(
            (2, _num_t * _num_states, _num_controls, _num_parameters))

    if quadrature.lower() == 'trapezoidal':

        def _fitting_function(_z: np.ndarray) -> np.ndarray:
            _t_0, _t_f = _z[0], _z[-1]
            _x = _z[_x_slice].reshape((_num_states, _num_t))
            _u = _z[_u_slice]
            _p = _z[_p_slice]

            _delta_t = (_t_f - _t_0)
            _t = _delta_t * _tau + _t_0

            res_bc = np.concatenate((
                _compute_initial_boundary_conditions(_t[0], _x[:, 0], _p, _k),
                _compute_terminal_boundary_conditions(_t[-1], _x[:, -1], _p, _k),
            ))

            _x_dot = np.array([
                _compute_dynamics(_t_i, _x_i, _u, _p, _k)
                for _t_i, _x_i in zip(_t, _x.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x - _h_arr * (_x_dot[:, :-1] + _x_dot[:, 1:]) / 2

            return np.concatenate((res_bc, res_dyn.flatten()))

    elif quadrature.lower() == 'midpoint':
        _tau_mid = (_tau[:-1] + _tau[1:]) / 2

        def _fitting_function(_z: np.ndarray) -> np.ndarray:
            _t_0, _t_f = _z[0], _z[-1]
            _x = _z[_x_slice].reshape((_num_states, _num_t))
            _u = _z[_u_slice]
            _p = _z[_p_slice]

            _delta_t = (_t_f - _t_0)
            _t = _delta_t * _tau + _t_0
            _t_mid = _delta_t * _tau_mid + _t_0

            _x_mid = (_x[:, :-1] + _x[:, 1:]) / 2

            res_bc = np.concatenate((
                _compute_initial_boundary_conditions(_t[0], _x[:, 0], _p, _k),
                _compute_terminal_boundary_conditions(_t[-1], _x[:, -1], _p, _k),
            ))

            _x_dot = np.array([
                _compute_dynamics(_t_i, _x_i, _u, _p, _k)
                for _t_i, _x_i in zip(_input_t, _x_mid.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x - _h_arr * _x_dot

            return np.concatenate((res_bc, res_dyn.flatten()))

    elif quadrature.lower() == 'simpson':
        _compute_dynamics = prob.compute_dynamics

        _tau_mid = (_tau[:-1] + _tau[1:]) / 2

        def _fitting_function(_z: np.ndarray) -> np.ndarray:
            _t_0, _t_f = _z[0], _z[-1]
            _x = _z[_x_slice].reshape((_num_states, _num_t))
            _u = _z[_u_slice]
            _p = _z[_p_slice]

            _delta_t = (_t_f - _t_0)
            _t = _delta_t * _tau + _t_0
            _t_mid = _delta_t * _tau_mid + _t_0

            res_bc = np.concatenate((
                _compute_initial_boundary_conditions(_t[0], _x[:, 0], _p, _k),
                _compute_terminal_boundary_conditions(_t[-1], _x[:, -1], _p, _k),
            ))

            _x_dot_nodes = np.array([
                _compute_dynamics(_t_i, _x_i, _u, _p, _k)
                for _t_i, _x_i in zip(_input_t, _x.T)
            ]).T

            _x_dot_diff = np.diff(_x_dot_nodes)
            _x_mid = (_x[:, :-1] + _x[:, 1:]) / 2 - _h_arr / 8 * np.diff(_x_dot_nodes)

            _x_dot_mid = np.array([
                _compute_dynamics(_t_i, _x_i, _u, _p, _k)
                for _t_i, _x_i in zip(_t_mid, _x_mid.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x \
                - _h_arr * ((_x_dot_nodes[:, :-1] + _x_dot_nodes[:, 1:]) / 6 + 2 / 3 * _x_dot_mid)

            return np.concatenate((res_bc, res_dyn.flatten()))

    else:
        raise ValueError(f'Quadrature {quadrature} not valid, must be \"trapezoidal\", \"midpoint\", or \"simpson\"')

    _matched = gauss_newton(
            _fitting_function, np.concatenate(
                    ((guess.t[0], guess.t[-1]), guess.x.flatten(), np.mean(guess.u, axis=1), guess.p)),
            rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose
    )

    guess.t = _tau * (_matched[-1] - _matched[0]) + _matched[0]
    guess.x = _matched[_x_slice].reshape((_num_states, _num_t))
    guess.u = np.broadcast_to(_matched[_u_slice], (_num_t, _num_controls)).T
    guess.p = _matched[_p_slice]

    return guess
