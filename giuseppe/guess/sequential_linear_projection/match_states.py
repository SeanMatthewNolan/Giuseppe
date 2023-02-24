from typing import Union
from copy import deepcopy

import numpy as np

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual
from .sequential_linear_projection import sequential_linearized_projection


def match_states(
        prob: Union[BVP, OCP, Dual], guess: Solution, quadrature: str = 'linear',
        rel_tol: float = 1e-4, abs_tol: float = 1e-4
) -> Solution:
    """

    Parameters
    ----------
    prob
    guess
    quadrature
    rel_tol
    abs_tol

    Returns
    -------

    """
    _num_states = prob.num_states
    _num_parameters = prob.num_parameters

    _compute_boundary_conditions = prob.compute_boundary_conditions

    if prob.prob_class == 'bvp':
        def _compute_dynamics(_t_i, _x_i, _, _p, _k):
            return prob.compute_dynamics(_t_i, _x_i, _p, _k)
    else:
        _compute_dynamics = prob.compute_dynamics

    guess = deepcopy(guess)
    _t, _u, _k = guess.t, guess.u, guess.k

    _num_t = len(_t)
    if _num_t < 2:
        raise RuntimeError('Please provide guess with at least 2 points')
    _h_arr = np.diff(_t)

    _x_slice = slice(_num_t * _num_states)
    _p_slice = slice(_x_slice.stop, _x_slice.stop + _num_parameters)

    if quadrature.lower() == 'linear':

        def _fitting_function(_states_and_parameters: np.ndarray) -> np.ndarray:
            _x = _states_and_parameters[_x_slice].reshape((_num_states, _num_t))
            _p = _states_and_parameters[_p_slice]

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
        _u_mid = (_u[:, :-1] + _u[:, 1:]) / 2

        def _fitting_function(_states_and_parameters: np.ndarray) -> np.ndarray:
            _x = _states_and_parameters[_x_slice].reshape((_num_states, _num_t))
            _p = _states_and_parameters[_p_slice]

            _x_mid = (_x[:, :-1] + _x[:, 1:]) / 2

            res_bc = _compute_boundary_conditions(_t, _x, _p, _k)

            _x_dot = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _u_i in zip(_t, _x_mid.T, _u_mid.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x - _h_arr * _x_dot

            return np.concatenate((res_bc, res_dyn.flatten()))

    elif quadrature.lower() == 'simpson':
        _compute_dynamics = prob.compute_dynamics

        _t_mid = (_t[:-1] + _t[1:]) / 2
        _u_mid = (_u[:, :-1] + _u[:, 1:]) / 2

        def _fitting_function(_states_and_parameters: np.ndarray) -> np.ndarray:
            _x = _states_and_parameters[_x_slice].reshape((_num_states, _num_t))
            _p = _states_and_parameters[_p_slice]

            res_bc = _compute_boundary_conditions(_t, _x, _p, _k)

            _x_dot_nodes = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _u_i in zip(_t, _x.T, _u.T)
            ]).T

            _x_dot_diff = np.diff(_x_dot_nodes)
            _x_mid = (_x[:, :-1] + _x[:, 1:]) / 2 - _h_arr / 8 * np.diff(_x_dot_nodes)

            _x_dot_mid = np.array([
                _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
                for _t_i, _x_i, _lam_i, _u_i in zip(_t_mid, _x_mid.T, _u_mid.T)
            ]).T

            _delta_x = np.diff(_x)

            res_dyn = _delta_x \
                - _h_arr * ((_x_dot_nodes[:, :-1] + _x_dot_nodes[:, 1:]) / 6 + 2 / 3 * _x_dot_mid)

            return np.concatenate((res_bc, res_dyn.flatten()))

    else:
        raise ValueError(f'Quadrature {quadrature} not valid, must be \"linear\", \"midpoint\", or \"simpson\"')

    adjoints = sequential_linearized_projection(
            _fitting_function, np.concatenate((guess.x.flatten(), guess.p)),
            rel_tol=rel_tol, abs_tol=abs_tol
    )

    guess.s = adjoints[_x_slice].reshape((_num_states, _num_t))
    guess.p = adjoints[_p_slice]

    return guess
