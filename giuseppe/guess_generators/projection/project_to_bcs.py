from copy import deepcopy
from typing import Union, Tuple, Optional

import numpy as np

from giuseppe.io.solution import Solution
from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, CompDual, \
    AdiffBVP, AdiffOCP, AdiffDual, AdiffDualOCP
from giuseppe.problems.dual.utils import sift_ocp_and_dual
from .project_to_nullspace import project_to_nullspace

SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP]


def match_constants_to_bcs(prob: SUPPORTED_PROBLEMS, guess: Solution,
                           rel_tol: float = 1e-3, abs_tol: float = 1e-3) -> Solution:
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
    elif isinstance(prob, AdiffBVP):
        def bc_func(k):
            psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.p, k)
            psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.p, k)
            return np.concatenate((np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten()))

    elif isinstance(prob, AdiffOCP):
        def bc_func(k):
            psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, k)
            psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, k)
            return np.concatenate((np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten()))

    elif isinstance(prob, AdiffDualOCP):
        ocp_bc = prob.ocp.ca_boundary_conditions
        ocp_bc = prob.ocp.ca_boundary_conditions
        dual_bc = prob.dual.ca_adj_boundary_conditions

        def bc_func(k):
            psi_0 = ocp_bc.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, k)
            psi_f = ocp_bc.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, k)

            adj_bc0 = dual_bc.initial(
                    guess.t[0], guess.x[:, 0], guess.lam[:, 0], guess.u[:, 0], guess.p, guess.nu0, k)
            adj_bcf = dual_bc.terminal(
                    guess.t[-1], guess.x[:, -1], guess.lam[:, -1], guess.u[:, -1], guess.p, guess.nuf, k)

            return np.concatenate((np.asarray(psi_0).flatten(),
                                   np.asarray(psi_f).flatten(),
                                   np.asarray(adj_bc0).flatten(),
                                   np.asarray(adj_bcf).flatten()))

    else:
        raise ValueError(f'Problem type {type(prob)} not supported')

    guess = deepcopy(guess)
    guess.k = project_to_nullspace(bc_func, guess.k, rel_tol=rel_tol, abs_tol=abs_tol)
    return guess


def match_states_to_bc(comp_prob: SUPPORTED_PROBLEMS, guess: Solution, location: str = 'initial',
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

    if location.lower() == 'initial':
        x_guess = guess.x[:, 0]
    elif location.lower() == 'terminal':
        x_guess = guess.x[:, -1]
    else:
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
    elif isinstance(prob, AdiffBVP):
        if location.lower() == 'initial':
            def bc_func(_x):
                psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], _x, guess.p, guess.k)
                return np.asarray(psi_0).flatten()

        else:
            def bc_func(_x):
                psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], _x, guess.p, guess.k)
                return np.asarray(psi_f).flatten()

    elif isinstance(prob, AdiffOCP):
        if location.lower() == 'initial':
            def bc_func(_x):
                psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], _x, guess.u[:, 0], guess.p, guess.k)
                return np.asarray(psi_0).flatten()

        else:
            def bc_func(_x):
                psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], _x, guess.u[:, -1], guess.p, guess.k)
                return np.asarray(psi_f).flatten()

    else:
        raise ValueError(f'Problem type {type(prob)} not supported')

    x = project_to_nullspace(bc_func, x_guess, rel_tol=rel_tol, abs_tol=abs_tol)

    if dual is not None and project_costates:
        lam = match_costates_to_bc(comp_prob, guess, location=location, states=x, rel_tol=rel_tol, abs_tol=abs_tol)
        return x, lam
    else:
        return x


def match_costates_to_bc(comp_prob: Union[CompDualOCP, CompDual], guess: Solution, location: str = 'initial',
                         states: Optional[np.ndarray] = None,
                         rel_tol: float = 1e-3, abs_tol: float = 1e-3) -> np.ndarray:
    """
    Projects the costates of a guess to the problem's boundary conditions to get the closest match

    Parameters
    ----------
    comp_prob : Solution
        problem whose BCs are to be matched
    guess : DualOCPSol
        guess from which to match the states (and costates)
    location : 'initial' or 'terminal', default='initial'
        specifies which BC to match
    states : np.ndarray, optional
        states at boundary to use
    abs_tol : float, default=1e-3
       absolute tolerance
    rel_tol : float, default=1e-3
       relative tolerance

    Returns
    -------
    projected costates

    """
    _, dual = sift_ocp_and_dual(comp_prob)

    if isinstance(dual, CompDual):
        dual_bc = dual.adjoined_boundary_conditions
    elif isinstance(dual, AdiffDual):
        dual_bc = dual.ca_adj_boundary_conditions
    else:
        raise ValueError(f'Problem type {type(comp_prob)} not supported')

    if location.lower() == 'initial':
        lam_guess = guess.lam[:, 0]

        if states is None:
            states = guess.x[:, 0]

        def adj_bc_func(_lam):
            adj_bc0 = dual_bc.initial(guess.t[0], states, _lam, guess.u[:, 0], guess.p, guess.nu0, guess.k)
            return np.asarray(adj_bc0).flatten()

    elif location.lower() == 'terminal':
        lam_guess = guess.lam[:, -1]
        if states is None:
            states = guess.x[:, -1]

        def adj_bc_func(_lam):
            adj_bcf = dual_bc.terminal(guess.t[-1], states, _lam, guess.u[:, -1], guess.p, guess.nuf, guess.k)
            return np.asarray(adj_bcf).flatten()

    else:
        raise ValueError(f'Location should be \'initial\' or \'terminal\', not {location}')

    lam = project_to_nullspace(adj_bc_func, lam_guess, rel_tol=rel_tol, abs_tol=abs_tol)

    return lam
