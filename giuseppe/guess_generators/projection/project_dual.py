from typing import Tuple, Union

import numpy as np
import casadi as ca

from giuseppe.utils.conversion import ca_vec2arr
from giuseppe.io.solution import Solution
from giuseppe.problems.dual import CompDualOCP, AdiffDualOCP
from .minimization_schemes import project_to_nullspace, gradient_descent, newtons_method

SUPPORTED_INPUTS = Union[CompDualOCP, AdiffDualOCP]


def project_dual(comp_prob: SUPPORTED_INPUTS, guess: Solution,
                 rel_tol: float = 1e-3, abs_tol: float = 1e-3, method: str = 'projection'):

    t = guess.t
    x = guess.x
    u = guess.u
    p = guess.p
    k = guess.k
    num_t = len(t)

    if isinstance(comp_prob, CompDualOCP):
        adjoined_bc_0 = comp_prob.comp_dual.adjoined_boundary_conditions.initial
        costate_dynamics = comp_prob.comp_dual.costate_dynamics
        adjoined_bc_f = comp_prob.comp_dual.adjoined_boundary_conditions.terminal

        num_nu0 = comp_prob.comp_dual.num_initial_adjoints
        num_lam = comp_prob.comp_dual.num_costates
        num_nuf = comp_prob.comp_dual.num_terminal_adjoints
    elif isinstance(comp_prob, AdiffDualOCP):
        adjoined_bc_0 = comp_prob.dual.ca_adj_boundary_conditions.initial
        costate_dynamics = comp_prob.dual.ca_costate_dynamics
        adjoined_bc_f = comp_prob.dual.ca_adj_boundary_conditions.terminal

        num_nu0 = comp_prob.dual.num_initial_adjoints
        num_lam = comp_prob.dual.num_costates
        num_nuf = comp_prob.dual.num_terminal_adjoints
    else:
        raise TypeError(f"comp_prob must be CompDualOCP or AdiffDualOCP; you used {type(comp_prob)}!")

    def unpack_values(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nu0 = values[:num_nu0]
        lam = np.reshape(values[num_nu0:num_nu0 + num_lam * num_t], (num_t, num_lam)).T
        nuf = values[num_nu0 + num_lam * num_t:num_nu0 + num_lam * num_t + num_nuf]
        return nu0, lam, nuf

    def residual(values: np.ndarray) -> np.ndarray:
        nu0, lam, nuf = unpack_values(values)
        if isinstance(adjoined_bc_0, ca.Function):
            bc_0 = ca_vec2arr(adjoined_bc_0(t[0], x[:, 0], lam[:, 0], u[:, 0], p, nu0, k))
            bc_f = ca_vec2arr(adjoined_bc_f(t[-1], x[:, -1], lam[:, -1], u[:, -1], p, nuf, k))
        else:
            bc_0 = np.asarray(adjoined_bc_0(t[0], x[:, 0], lam[:, 0], u[:, 0], p, nu0, k)).flatten()
            bc_f = np.asarray(adjoined_bc_f(t[-1], x[:, -1], lam[:, -1], u[:, -1], p, nuf, k)).flatten()
        dyn_res = []
        for idx in range(num_t - 1):
            t_left, t_right = t[idx], t[idx + 1]
            x_left, x_right = x[:, idx], x[:, idx + 1]
            lam_left, lam_right = lam[:, idx], lam[:, idx + 1]
            u_left, u_right = u[:, idx], u[:, idx + 1]

            dt = t_right - t_left
            t_bar = (t_right + t_left) / 2
            x_bar = (x_right + x_left) / 2
            lam_bar = (lam_right + lam_left) / 2
            u_bar = (u_right + u_left) / 2

            d_lam = costate_dynamics(t_bar, x_bar, lam_bar, u_bar, p, k)
            if isinstance(d_lam, Union[ca.SX, ca.DM]):
                d_lam = ca_vec2arr(d_lam)
            else:
                d_lam = np.asarray(d_lam).flatten()
            dyn_res.append(
                lam_right - lam_left - dt * d_lam)

        return np.concatenate((bc_0, np.array(dyn_res).flatten(), bc_f))

    adj_vars_guess = np.concatenate((guess.nu0, guess.lam.T.flatten(), guess.nuf))
    if method == 'projection':
        adj_vars_optimized = project_to_nullspace(residual, adj_vars_guess, rel_tol=rel_tol, abs_tol=abs_tol)
    elif method == 'gradient':
        adj_vars_optimized = gradient_descent(residual, adj_vars_guess, abs_tol=abs_tol)
    elif method == 'newton':
        adj_vars_optimized = gradient_descent(residual, adj_vars_guess, abs_tol=abs_tol)
    else:
        raise(RuntimeError, f'Optimization Method invalid!'
                            f'Should be:\nprojection\ngradient\nnewton\nYou used:\n{method}')
    guess.nu0, guess.lam, guess.nuf = unpack_values(adj_vars_optimized)

    return guess
