from copy import deepcopy
from typing import Union, Tuple

import numpy as np

from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, BVPSol, OCPSol, DualOCPSol
from giuseppe.problems.dual.utils import sift_ocp_and_dual
from .project_to_nullspace import project_to_nullspace

SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP]
SUPPORTED_SOLUTIONS = Union[BVPSol, OCPSol, DualOCPSol]


def match_constants_to_bcs(prob: SUPPORTED_PROBLEMS, guess: SUPPORTED_SOLUTIONS,
                           rel_tol: float = 1e-3, abs_tol: float = 1e-3) -> SUPPORTED_SOLUTIONS:
    """
    Projects the constant array of a guess to the problem's boundary conditions to get the closest match

    Parameters
    ----------
    prob : CompBVP or CompOCP or CompDualOCP
        problem whose BCs are to be matched
    guess : BVPSol or OCPSol or DualOCPSol
        guess from which to match the constants
    abs_tol : float, default=1e-3
       absolute tolerance
    rel_tol : float, default=1e-3
       relative tolerance

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
    guess.k = project_to_nullspace(bc_func, guess.k, rel_tol=rel_tol, abs_tol=abs_tol)
    return guess


def match_states_to_bc(comp_prob: SUPPORTED_PROBLEMS, guess: SUPPORTED_SOLUTIONS, location: str = 'initial',
                       project_costates: bool = False, rel_tol: float = 1e-3, abs_tol: float = 1e-3
                       ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Projects the state (and costates) of a guess to the problem's boundary conditions to get the closest match

    Parameters
    ----------
    comp_prob : CompBVP or CompOCP or CompDualOCP
        problem whose BCs are to be matched
    guess : BVPSol or OCPSol or DualOCPSol
        guess from which to match the states (and costates)
    location : 'initial' or 'terminal', default='initial'
        specifies which BC to match
    project_costates : bool, default=False
        specifies whether to project the costates as well as the states
    abs_tol : float, default=1e-3
       absolute tolerance
    rel_tol : float, default=1e-3
       relative tolerance

    Returns
    -------
    projected states (and costates)

    """
    prob, dual = sift_ocp_and_dual(comp_prob)

    if location.lower() not in ['initial', 'terminal']:
        raise ValueError(f'Location should be \'initial\' or \'terminal\', not {location}')

    if isinstance(prob, CompBVP):
        if location.lower() == 'initial':
            def bc_func(_x):
                psi_0 = prob.boundary_conditions.initial(guess.t[0], _x, guess.p, guess.k)
                return np.asarray(psi_0)

        else:
            def bc_func(_x):
                psi_f = prob.boundary_conditions.terminal(guess.t[-1], _x, guess.p, guess.k)
                return np.asarray(psi_f)

    elif isinstance(prob, CompOCP):
        if location.lower() == 'initial':
            def bc_func(_x):
                psi_0 = prob.boundary_conditions.initial(guess.t[0], _x, guess.u[:, 0], guess.p, guess.k)
                return np.asarray(psi_0)

        else:
            def bc_func(_x):
                psi_f = prob.boundary_conditions.terminal(guess.t[-1], _x, guess.u[:, -1], guess.p, guess.k)
                return np.asarray(psi_f)

    else:
        raise ValueError(f'Problem type {type(prob)} not supported')

    x = project_to_nullspace(bc_func, guess.x, rel_tol=rel_tol, abs_tol=abs_tol)

    if dual is not None and project_costates:
        dual_bc = dual.adjoined_boundary_conditions

        if location.lower() == 'initial':
            def adj_bc_func(_lam):
                adj_bc0 = dual_bc.initial(
                    guess.t[0], x, _lam, guess.u[:, 0], guess.p, guess.nu0, guess.k)
                return np.asarray(adj_bc0)
        else:
            def adj_bc_func(_lam):
                adj_bcf = dual_bc.terminal(
                    guess.t[-1], x, _lam, guess.u[:, -1], guess.p, guess.nuf, guess.k)
                return np.asarray(adj_bcf)

        lam = project_to_nullspace(adj_bc_func, guess.lam, rel_tol=rel_tol, abs_tol=abs_tol)

        return x, lam

    else:
        return x
