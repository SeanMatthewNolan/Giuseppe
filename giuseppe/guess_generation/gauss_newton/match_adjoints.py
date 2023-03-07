from copy import deepcopy

import numpy as np

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import Dual
from giuseppe.utils import make_array_slices
from .gauss_newton import gauss_newton


def match_adjoints(
        prob: Dual, guess: Solution, quadrature: str = 'trapezoidal', rel_tol: float = 1e-4, abs_tol: float = 1e-4,
        condition_adjoints: bool = False, verbose: bool = False
) -> Solution:

    _num_costates = prob.num_costates
    _num_adjoints = prob.num_adjoints

    _compute_adjoint_boundary_conditions = prob.compute_adjoint_boundary_conditions
    _compute_costate_dynamics = prob.compute_costate_dynamics
    _compute_hamiltonian = prob.compute_hamiltonian

    guess = deepcopy(guess)
    _t, _x, _u, _p, _k = guess.t, guess.x, guess.u, guess.p, guess.k

    _num_t = len(_t)
    if _num_t < 2:
        raise RuntimeError('Please provide guess with at least 2 points')
    _h_arr = np.diff(_t)

    if condition_adjoints:
        _dx = np.diff(_x)
        _dx_dt = _dx / _h_arr
        conditioning = np.fmax(np.mean(np.abs(_dx_dt), axis=1), rel_tol)
        rel_tol = 1.

    _lam_slice, _nu_slice = make_array_slices((_num_t * _num_costates, _num_adjoints))

    if quadrature.lower() == 'trapezoidal':

        def _fitting_function(_adjoints: np.ndarray) -> np.ndarray:
            _lam = _adjoints[_lam_slice].reshape((_num_costates, _num_t))
            _nu = _adjoints[_nu_slice]

            res_bc = _compute_adjoint_boundary_conditions(_t, _x, _lam, _u, _p, _nu, _k)

            _lam_dot = np.array([
                _compute_costate_dynamics(_t_i, _x_i, _lam_i, _u_i, _p, _k)
                for _t_i, _x_i, _lam_i, _u_i in zip(_t, _x.T, _lam.T, _u.T)
            ]).T

            _delta_lam = np.diff(_lam)

            res_dyn = _delta_lam - _h_arr * (_lam_dot[:, :-1] + _lam_dot[:, 1:]) / 2

            if condition_adjoints:
                res_dyn = res_dyn.T * conditioning

            return np.concatenate((res_bc, res_dyn.flatten()))

    elif quadrature.lower() == 'midpoint':
        _t_mid = (_t[:-1] + _t[1:]) / 2
        _x_mid = (_x[:, :-1] + _x[:, 1:]) / 2
        _u_mid = (_u[:, :-1] + _u[:, 1:]) / 2

        def _fitting_function(_adjoints: np.ndarray) -> np.ndarray:
            _lam = _adjoints[_lam_slice].reshape((_num_costates, _num_t))
            _nu = _adjoints[_nu_slice]

            _lam_mid = (_lam[:, :-1] + _lam[:, 1:]) / 2

            res_bc = _compute_adjoint_boundary_conditions(_t, _x, _lam, _u, _p, _nu, _k)

            _lam_dot = np.array([
                _compute_costate_dynamics(_t_i, _x_i, _lam_i, _u_i, _p, _k)
                for _t_i, _x_i, _lam_i, _u_i in zip(_t, _x_mid.T, _lam_mid.T, _u_mid.T)
            ]).T

            _delta_lam = np.diff(_lam)

            res_dyn = _delta_lam - _h_arr * _lam_dot

            if condition_adjoints:
                res_dyn = res_dyn.T * conditioning

            return np.concatenate((res_bc, res_dyn.flatten()))

    elif quadrature.lower() == 'simpson':
        _compute_dynamics = prob.compute_dynamics

        _t_mid = (_t[:-1] + _t[1:]) / 2
        _u_mid = (_u[:, :-1] + _u[:, 1:]) / 2

        _x_dot = np.array([
            _compute_dynamics(_t_i, _x_i, _u_i, _p, _k)
            for _t_i, _x_i, _u_i in zip(_t, _x.T, _u.T)
        ]).T
        _x_mid = (_x[:, :-1] + _x[:, 1:]) / 2 - _h_arr / 8 * np.diff(_x_dot)

        def _fitting_function(_adjoints: np.ndarray) -> np.ndarray:
            _lam = _adjoints[_lam_slice].reshape((_num_costates, _num_t))
            _nu = _adjoints[_nu_slice]

            res_bc = _compute_adjoint_boundary_conditions(_t, _x, _lam, _u, _p, _nu, _k)

            _lam_dot_nodes = np.array([
                _compute_costate_dynamics(_t_i, _x_i, _lam_i, _u_i, _p, _k)
                for _t_i, _x_i, _lam_i, _u_i in zip(_t, _x.T, _lam.T, _u.T)
            ]).T

            _lam_dot_diff = np.diff(_lam_dot_nodes)
            _lam_mid = (_lam[:, :-1] + _lam[:, 1:]) / 2 - _h_arr / 8 * np.diff(_lam_dot_nodes)

            _lam_dot_mid = np.array([
                _compute_costate_dynamics(_t_i, _x_i, _lam_i, _u_i, _p, _k)
                for _t_i, _x_i, _lam_i, _u_i in zip(_t_mid, _x_mid.T, _lam_mid.T, _u_mid.T)
            ]).T

            _delta_lam = np.diff(_lam)

            res_dyn = _delta_lam \
                - _h_arr * ((_lam_dot_nodes[:, :-1] + _lam_dot_nodes[:, 1:]) / 6 + 2 / 3 * _lam_dot_mid)

            if condition_adjoints:
                res_dyn = res_dyn.T * conditioning

            return np.concatenate((res_bc, res_dyn.flatten()))

    else:
        raise ValueError(f'Quadrature {quadrature} not valid, must be \"trapezoidal\", \"midpoint\", or \"simpson\"')

    adjoints = gauss_newton(
            _fitting_function, np.concatenate((guess.lam.flatten(), guess.nu0, guess.nuf)),
            rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose
    )

    guess.lam = adjoints[_lam_slice].reshape((_num_costates, _num_t))
    guess.nu0 = adjoints[_nu_slice][:prob.num_initial_adjoints]
    guess.nuf = adjoints[_nu_slice][prob.num_initial_adjoints:]

    return guess
