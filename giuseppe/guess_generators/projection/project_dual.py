from typing import Tuple, Union

import numpy as np

from giuseppe.problems.dual import CompDualOCP, AdiffDualOCP, DualOCPSol
from . import project_to_nullspace

SUPPORTED_INPUTS = Union[CompDualOCP, AdiffDualOCP]


def project_dual(comp_prob: SUPPORTED_INPUTS, guess: DualOCPSol, rel_tol: float = 1e-3, abs_tol: float = 1e-3):

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

            dyn_res.append(lam_right - lam_left - dt * np.asarray(costate_dynamics(t_bar, x_bar, lam_bar, u_bar, p, k)).flatten())

        return np.concatenate((bc_0, np.array(dyn_res).flatten(), bc_f))

    adj_vars_guess = np.concatenate((guess.nu0, guess.lam.T.flatten(), guess.nuf))
    guess.nu0, guess.lam, guess.nuf = unpack_values(
            project_to_nullspace(residual, adj_vars_guess, rel_tol=rel_tol, abs_tol=abs_tol))

    return guess
