from copy import deepcopy

import numpy as np

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import Dual
from .sequential_linear_projection import sequential_linearized_projection


def match_adjoints(
        prob: Dual, guess: Solution, quadrature: str = 'linear', rel_tol: float = 1e-4, abs_tol: float = 1e-4
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

    _lam_slice = slice(_num_t * _num_costates)
    _nu_slice = slice(_lam_slice.stop, _lam_slice.stop + _num_adjoints)

    if quadrature.lower() == 'linear':

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

            _lam_mid = (_lam[:, :-1] + _lam[:, 1:]) / 2

            res_bc = _compute_adjoint_boundary_conditions(_t, _x, _lam, _u, _p, _nu, _k)

            _lam_dot_nodes = np.array([
                _compute_costate_dynamics(_t_i, _x_i, _lam_i, _u_i, _p, _k)
                for _t_i, _x_i, _lam_i, _u_i in zip(_t, _x.T, _lam.T, _u.T)
            ]).T

            _lam_dot_diff = np.diff(_lam_dot_nodes)
            _lam_mid = _lam_mid - _h_arr / 8 * np.diff(_lam_dot_nodes)

            _lam_dot_mid = np.array([
                _compute_costate_dynamics(_t_i, _x_i, _lam_i, _u_i, _p, _k)
                for _t_i, _x_i, _lam_i, _u_i in zip(_t_mid, _x_mid.T, _lam_mid.T, _u_mid.T)
            ]).T

            _delta_lam = np.diff(_lam)

            res_dyn = _delta_lam \
                - _h_arr * ((_lam_dot_nodes[:, :-1] + _lam_dot_nodes[:, 1:]) / 6 + 2 / 3 * _lam_dot_mid)

            return np.concatenate((res_bc, res_dyn.flatten()))
    else:
        raise ValueError(f'Quadrature {quadrature} not valid, must be \"linear\", \"midpoint\", or \"simpson\"')

    adjoints = sequential_linearized_projection(
            _fitting_function, np.concatenate((guess.lam.flatten(), guess.nu0, guess.nuf)),
            rel_tol=rel_tol, abs_tol=abs_tol
    )

    guess.lam = adjoints[_lam_slice].reshape((_num_costates, _num_t))
    guess.nu0 = adjoints[_nu_slice][:prob.num_initial_adjoints]
    guess.nuf = adjoints[_nu_slice][prob.num_initial_adjoints:]

    return guess
