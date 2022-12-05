from typing import Union, Optional

import numpy as np
import casadi as ca
from numpy.typing import ArrayLike

from giuseppe.io.solution import Solution
from giuseppe.problems import CompBVP, CompOCP, CompDual, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDual, AdiffDualOCP
from giuseppe.problems.dual.utils import sift_ocp_and_dual
from giuseppe.utils.conversion import ca_vec2arr
from .simple import update_linear_value
from ..constant import initialize_guess_for_auto, update_constant_value
from ..projection import project_to_nullspace, gradient_descent, newtons_method

SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP]


def auto_linear_guess(comp_prob: SUPPORTED_PROBLEMS, t_span: Union[float, ArrayLike] = 0.1,
                      constants: Optional[ArrayLike] = None, default: Union[float, Solution] = 0.1,
                      use_dynamics: bool = False, propagation_method: bool = 'rk',
                      abs_tol: float = 1e-3, rel_tol: float = 1e-3, backtrack: bool = True,
                      method: str = 'projection') -> Solution:
    """
    Automatically generate guess where variables (excluding the indenpendent) with initial and terminal values fitted
    to the boundary conditions and dynamics

    Parameters
    ----------
    comp_prob : CompBVP, CompOCP, or CompDualOCP 
        the problem that the guess is for
    t_span : float or ArrayLike, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])
    constants : ArrayLike, optional
        constant  values
    default : float, BVPSol, OCPSol, or DualOCPSol, default=0.1
        if float, default value for set variables
        if solution type, guess to modify from which unmodified values will be taken
    use_dynamics : bool, default=True
        specifies whether the values are fitted with respect to a linear approximation of the dynamics or BC alone
    propagation_method : str, default='rk'
        Propagation method to fit to dynamics. Supported inputs are: euler, rk
    abs_tol : float, default=1e-3
        absolute tolerance
    rel_tol : float, default=1e-3
        relative tolerance
    backtrack : bool, default=True
        Whether to use backtracking line search during minimization
    method : str, default='projection'
        Optimization method to minimize residual. Supported inputs are: projection, gradient, newton

    Returns
    -------
    guess : Solution

    """
    prob, dual = sift_ocp_and_dual(comp_prob)
    guess = initialize_guess_for_auto(comp_prob, t_span=t_span, constants=constants, default=default)

    # Project states, (controls, ) and parameters to BCs
    num_x = prob.num_states
    num_p = prob.num_parameters

    if method == 'projection':
        def optimize(func, arr):
            return project_to_nullspace(func, arr, abs_tol=abs_tol, rel_tol=rel_tol, backtrack=backtrack)
    elif method == 'gradient':
        def optimize(func, arr):
            return gradient_descent(func, arr, abs_tol=abs_tol, backtrack=backtrack)
    elif method == 'newton':
        def optimize(func, arr):
            return newtons_method(func, arr, abs_tol=abs_tol, backtrack=backtrack)
    else:
        raise(RuntimeError, f'Optimization Method invalid!'
                            f'Should be:\nprojection\ngradient\nnewton\nYou used:\n{method}')

    if propagation_method == 'euler':
        def difference_func(dyn_func, step_size, args_right, args_left, args_middle):
            return step_size * dyn_func(*args_middle)
    elif propagation_method == 'rk':
        def difference_func(dyn_func, step_size, args_right, args_left, args_middle):
            return step_size / 6.0 * (dyn_func(*args_left) + 4 * dyn_func(*args_middle) + dyn_func(*args_right))
    else:
        raise(RuntimeError, f'Propogation method invalid!'
                            f'Should be:\neuler\nrk\nYou used:\n{propagation_method}')

    if isinstance(prob, CompBVP) or isinstance(prob, AdiffBVP):
        def map_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            _x0 = values[:num_x]
            _xf = values[num_x:2 * num_x]
            _p = values[2 * num_x:2 * num_x + num_p]
            return _x0, _xf, _p

        if isinstance(prob, CompBVP):
            boundary_conditions = prob.boundary_conditions
            dynamics = prob.dynamics
        else:
            boundary_conditions = prob.ca_boundary_conditions
            dynamics = prob.ca_dynamics

        if use_dynamics:
            def bc_func(values):
                t_left, t_right = guess.t[0], guess.t[-1]
                x_left, x_right, _p = map_values(values)
                dt = t_right - t_left
                t_bar = 0.5 * (t_left + t_right)
                x_bar = 0.5 * (x_left + x_right)
                dx = difference_func(dynamics, dt,
                                     (t_right, x_right, _p, guess.k),
                                     (t_left, x_left, _p, guess.k),
                                     (t_bar, x_bar, _p, guess.k))

                if isinstance(dx, Union[ca.SX, ca.DM]):
                    dx = ca_vec2arr(dx)
                else:
                    dx = np.asarray(dx).flatten()

                psi_0 = boundary_conditions.initial(t_left, x_left, _p, guess.k)
                dyn_res = _xf - _x0 - dx
                psi_f = boundary_conditions.terminal(t_right, x_right, _p, guess.k)
                return np.concatenate((np.asarray(psi_0).flatten(), dyn_res, np.asarray(psi_f).flatten()))

        else:
            def bc_func(values):
                _x0, _xf, _p = map_values(values)
                psi_0 = boundary_conditions.initial(guess.t[0], _x0, _p, guess.k)
                psi_f = boundary_conditions.terminal(guess.t[-1], _xf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten()))

        values_guess = np.concatenate(guess.x[:, 0], guess.x[:, -1], guess.p)
        out_x0, out_xf, out_p = map_values(optimize(bc_func, values_guess))
        update_linear_value(guess, 'x', out_x0, out_xf)
        update_constant_value(guess, 'p', out_p)

    elif isinstance(prob, CompOCP) or isinstance(prob, AdiffOCP):

        num_u = prob.num_controls

        if isinstance(prob, CompOCP):
            dynamics = prob.dynamics
        else:
            dynamics = prob.ca_dynamics

        def map_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            _x0 = values[:num_x]
            _xf = values[num_x:2 * num_x]
            _u0 = values[2 * num_x:2 * num_x + num_u]
            _uf = values[2 * num_x + num_u:2 * (num_x + num_u)]
            _p = values[2 * (num_x + num_u):2 * (num_x + num_u) + num_p]
            return _x0, _xf, _u0, _uf, _p

        if isinstance(prob, CompOCP):
            boundary_conditions = prob.boundary_conditions
        else:
            boundary_conditions = prob.ca_boundary_conditions

        if use_dynamics:
            def bc_func(values):
                x_left, x_right, u_left, u_right, _p = map_values(values)
                t_left, t_right = guess.t[0], guess.t[-1]
                dt = t_right - t_left
                t_bar = 0.5 * (t_left + t_right)
                x_bar = 0.5 * (x_left + x_right)
                u_bar = 0.5 * (u_left + u_right)

                psi_0 = boundary_conditions.initial(t_left, x_left, u_left, _p, guess.k)
                dx = difference_func(dynamics, dt,
                                     (t_right, x_right, u_right, _p, guess.k),
                                     (t_left, x_left, u_left, _p, guess.k),
                                     (t_bar, x_bar, u_bar, _p, guess.k))
                if isinstance(dx, Union[ca.SX, ca.DM]):
                    dx = ca_vec2arr(dx)
                else:
                    dx = np.asarray(dx).flatten()

                dyn_res = x_right - x_left - dx
                psi_f = boundary_conditions.terminal(t_right, x_right, u_right, _p, guess.k)
                return np.concatenate((np.asarray(psi_0).flatten(), dyn_res, np.asarray(psi_f).flatten()))

        else:
            def bc_func(values):
                _x0, _xf, _u0, _uf, _p = map_values(values)
                psi_0 = boundary_conditions.initial(guess.t[0], _x0, _u0, _p, guess.k)
                psi_f = boundary_conditions.terminal(guess.t[-1], _xf, _uf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten()))

        values_guess = np.concatenate((guess.x[:, 0], guess.x[:, -1], guess.u[:, 0], guess.u[:, -1], guess.p))
        out_x0, out_xf, out_u0, out_uf, out_p = map_values(
                optimize(bc_func, values_guess))
        update_linear_value(guess, 'x', out_x0, out_xf)
        update_linear_value(guess, 'u', out_u0, out_uf)
        update_constant_value(guess, 'p', out_p)

        if dual is not None:

            num_lam = dual.num_costates
            num_nu0 = dual.num_initial_adjoints
            num_nuf = dual.num_terminal_adjoints

            ocp_bc = boundary_conditions

            if isinstance(dual, CompDual):
                dual_bc = dual.adjoined_boundary_conditions
            else:
                dual_bc = dual.ca_adj_boundary_conditions

            def map_dual_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                _x0 = values[:num_x]
                _xf = values[num_x:2 * num_x]
                _lam0 = values[2 * num_x:2 * num_x + num_lam]
                _lamf = values[2 * num_x + num_lam:2 * (num_x + num_lam)]
                _u0 = values[2 * (num_x + num_lam):2 * (num_x + num_lam) + num_u]
                _uf = values[2 * (num_x + num_lam) + num_u:2 * (num_x + num_lam + num_u)]
                _p = values[2 * (num_x + num_lam + num_u):num_x + num_lam + num_u + num_p]
                _nu0 = values[2 * (num_x + num_lam + num_u) + num_p:2 * (num_x + num_lam + num_u) + num_p + num_nu0]
                _nuf = values[2 * (num_x + num_lam + num_u) + num_p + num_nu0:
                              2 * (num_x + num_lam + num_u) + num_p + num_nu0 + num_nuf]
                return _x0, _xf, _lam0, _lamf, _u0, _uf, _p, _nu0, _nuf

            if use_dynamics:
                def dual_bc_func(values):
                    x_left, x_right, lam_left, lam_right, u_left, u_right, _p, _nu0, _nuf = map_dual_values(values)
                    t_left, t_right = guess.t[0], guess.t[-1]
                    dt = t_right - t_left
                    t_bar = 0.5 * (t_left + t_right)
                    x_bar = 0.5 * (x_left + x_right)
                    lam_bar = 0.5 * (lam_left + lam_right)
                    u_bar = 0.5 * (u_left + u_right)

                    psi_0 = ocp_bc.initial(t_left, x_left, u_left, _p, guess.k)

                    dx = difference_func(dynamics, dt,
                                         (t_right, x_right, u_right, _p, guess.k),
                                         (t_left, x_left, u_left, _p, guess.k),
                                         (t_bar, x_bar, u_bar, _p, guess.k))
                    if isinstance(dx, Union[ca.SX, ca.DM]):
                        dx = ca_vec2arr(dx)
                    else:
                        dx = np.asarray(dx).flatten()

                    dyn_res = _xf - _x0 - dx
                    psi_f = ocp_bc.terminal(t_right, x_right, u_right, _p, guess.k)

                    adj_bc0 = dual_bc.initial(t_left, x_left, lam_left, u_left, _p, _nu0, guess.k)
                    d_lam = difference_func(dual.costate_dynamics, dt,
                                            (t_right, x_right, lam_right, u_right, _p, guess.k),
                                            (t_left, x_left, lam_left, u_left, _p, guess.k),
                                            (t_bar, x_bar, lam_bar, u_bar, _p, guess.k))
                    if isinstance(d_lam, Union[ca.SX, ca.DM]):
                        d_lam = ca_vec2arr(d_lam)
                    else:
                        d_lam = np.asarray(d_lam).flatten()

                    costate_dyn_res = _lamf - _lam0 - d_lam
                    adj_bcf = dual_bc.terminal(t_right, x_right, lam_right, u_right, _p, _nuf, guess.k)

                    return np.concatenate((
                        np.asarray(psi_0).flatten(), dyn_res, np.asarray(psi_f).flatten(),
                        np.asarray(adj_bc0).flatten(), costate_dyn_res, np.asarray(adj_bcf).flatten()
                    ))

            else:
                def dual_bc_func(values):
                    _x0, _xf, _lam0, _lamf, _u0, _uf, _p, _nu0, _nuf = map_dual_values(values)

                    psi_0 = ocp_bc.initial(guess.t[0], _x0, _u0, _p, guess.k)
                    psi_f = ocp_bc.terminal(guess.t[-1], _xf, _uf, _p, guess.k)

                    adj_bc0 = dual_bc.initial(guess.t[0], _x0, _lam0, _u0, _p, _nu0, guess.k)
                    adj_bcf = dual_bc.terminal(guess.t[-1], _xf, _lamf, _uf, _p, _nuf, guess.k)

                    return np.concatenate((
                        np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten(),
                        np.asarray(adj_bc0).flatten(), np.asarray(adj_bcf).flatten()
                    ))

            values_guess = np.concatenate(
                    (guess.x[:, 0], guess.lam[:, 0], guess.u[:, 0], guess.x[:, -1], guess.lam[:, -1], guess.u[:, -1],
                     guess.p, guess.nu0, guess.nuf)
            )
            out_x0, out_xf, out_lam0, out_lamf, out_u0, out_uf, out_p, out_nu0, out_nuf = map_dual_values(
                    optimize(dual_bc_func, values_guess)
            )
            update_linear_value(guess, 'x', out_x0, out_xf)
            update_linear_value(guess, 'lam', out_lam0, out_lamf)
            update_linear_value(guess, 'u', out_u0, out_uf)
            update_constant_value(guess, 'p', out_p)
            update_constant_value(guess, 'nu0', out_nu0)
            update_constant_value(guess, 'nuf', out_nuf)
    else:
        raise ValueError(f'Problem type {type(comp_prob)} not supported')

    return guess
