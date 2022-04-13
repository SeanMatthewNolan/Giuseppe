from copy import deepcopy
from typing import Union

import numpy as np

from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, BVPSol, OCPSol, DualOCPSol
from .project_to_nullspace import project_to_nullspace

SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP]
SUPPORTED_SOLUTIONS = Union[BVPSol, OCPSol, DualOCPSol]


def match_constants_to_bcs(prob: SUPPORTED_PROBLEMS, guess: SUPPORTED_SOLUTIONS) -> SUPPORTED_SOLUTIONS:
    """
    Projects the constant array of a guess to the problem's boundary conditions to get the closest match

    Parameters
    ----------
    prob : CompBVP or CompOCP or CompDualOCP
        problem whose BCs are to be matched
    guess : BVPSol or OCPSol or DualOCPSol
        guess from which to match the constants

    Returns
    -------
    guess with projected constants

    """
    if isinstance(prob, CompBVP):
        def bc_func(k):
            psi_0 = prob.boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.p, k)
            psi_f = prob.boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.p, k)
            return np.concatenate((np.asarray(psi_0), np.asarray(psi_f)))

    elif isinstance(prob, CompOCP):
        def bc_func(k):
            psi_0 = prob.boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, k)
            psi_f = prob.boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, k)
            return np.concatenate((np.asarray(psi_0), np.asarray(psi_f)))

    elif isinstance(prob, CompDualOCP):
        ocp_bc = prob.comp_ocp.boundary_conditions
        dual_bc = prob.comp_dual.adjoined_boundary_conditions

        def bc_func(k):
            psi_0 = ocp_bc.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, k)
            psi_f = ocp_bc.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, k)

            adj_bc0 = dual_bc.initial(
                    guess.t[0], guess.x[:, 0], guess.lam[:, 0], guess.u[:, 0], guess.p, guess.nu0, k)
            adj_bcf = dual_bc.terminal(
                    guess.t[-1], guess.x[:, -1], guess.lam[:, -1], guess.u[:, -1], guess.p, guess.nuf, k)

            return np.concatenate((np.asarray(psi_0), np.asarray(psi_f), np.asarray(adj_bc0), np.asarray(adj_bcf)))

    else:
        raise ValueError(f'Problem type {type(prob)} not supported')

    guess = deepcopy(guess)
    guess.k = project_to_nullspace(bc_func, guess.k)
    return guess
