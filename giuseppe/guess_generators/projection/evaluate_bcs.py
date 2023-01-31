from __future__ import annotations
import sys
from typing import Union, Callable, Tuple

import numpy as np

from giuseppe.data_classes.solution import Solution
from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP


SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP]


def generate_bc_func(prob: SUPPORTED_PROBLEMS) -> Callable[[Solution], np.ndarray]:
    """
    Generates BC Functions to Judge Validity of Guess

    Parameters
    ----------
    prob : CompBVP or CompOCP or CompDualOCP
        problem whose BCs are to be matched

    Returns
    -------
    BC Func

    """
    if isinstance(prob, CompBVP):
        def bc_func(guess):
            psi_0 = prob.boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.p, guess.k)
            psi_f = prob.boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.p, guess.k)
            return np.concatenate((np.asarray(psi_0), np.asarray(psi_f)))

    elif isinstance(prob, CompOCP):
        def bc_func(guess):
            psi_0 = prob.boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, guess.k)
            psi_f = prob.boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, guess.k)
            return np.concatenate((np.asarray(psi_0), np.asarray(psi_f)))

    elif isinstance(prob, CompDualOCP):
        ocp_bc = prob.comp_ocp.boundary_conditions
        dual_bc = prob.comp_dual.adjoined_boundary_conditions

        def bc_func(guess):
            psi_0 = ocp_bc.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, guess.k)
            psi_f = ocp_bc.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, guess.k)

            adj_bc0 = dual_bc.initial(
                    guess.t[0], guess.x[:, 0], guess.lam[:, 0], guess.u[:, 0], guess.p, guess.nu0, guess.k)
            adj_bcf = dual_bc.terminal(
                    guess.t[-1], guess.x[:, -1], guess.lam[:, -1], guess.u[:, -1], guess.p, guess.nuf, guess.k)

            return np.concatenate((np.asarray(psi_0), np.asarray(psi_f), np.asarray(adj_bc0), np.asarray(adj_bcf)))
    elif isinstance(prob, AdiffBVP):
        def bc_func(guess):
            psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.p, guess.k)
            psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.p, guess.k)
            return np.concatenate((np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten()))

    elif isinstance(prob, AdiffOCP):
        def bc_func(guess):
            psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, guess.k)
            psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, guess.k)
            return np.concatenate((np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten()))

    elif isinstance(prob, AdiffDualOCP):
        ocp_bc = prob.ocp.ca_boundary_conditions
        ocp_bc = prob.ocp.ca_boundary_conditions
        dual_bc = prob.dual.ca_adj_boundary_conditions

        def bc_func(guess):
            psi_0 = ocp_bc.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, guess.k)
            psi_f = ocp_bc.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, guess.k)

            adj_bc0 = dual_bc.initial(
                    guess.t[0], guess.x[:, 0], guess.lam[:, 0], guess.u[:, 0], guess.p, guess.nu0, guess.k)
            adj_bcf = dual_bc.terminal(
                    guess.t[-1], guess.x[:, -1], guess.lam[:, -1], guess.u[:, -1], guess.p, guess.nuf, guess.k)

            return np.concatenate((np.asarray(psi_0).flatten(),
                                   np.asarray(psi_f).flatten(),
                                   np.asarray(adj_bc0).flatten(),
                                   np.asarray(adj_bcf).flatten()))

    else:
        raise ValueError(f'Problem type {type(prob)} not supported')

    return bc_func


if sys.version_info >= (3, 10):
    SEPARATED_BCS = tuple[Callable[[Solution], np.ndarray], Callable[[Solution], np.ndarray]]
else:
    SEPARATED_BCS = Tuple[Callable, Callable]


def generate_separated_bc_funcs(prob: SUPPORTED_PROBLEMS) -> SEPARATED_BCS:
    """
    Generates BC Functions to Judge Validity of Guess

    Parameters
    ----------
    prob : CompBVP or CompOCP or CompDualOCP
        problem whose BCs are to be matched

    Returns
    -------
    Initial BC Function, Terminal BC Function

    """
    if isinstance(prob, CompBVP):
        def bc0_func(guess):
            return np.asarray(prob.boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.p, guess.k))

        def bcf_func(guess):
            return np.asarray(prob.boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.p, guess.k))

    elif isinstance(prob, CompOCP):
        def bc0_func(guess):
            return np.asarray(
                    prob.boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, guess.k))

        def bcf_func(guess):
            return np.asarray(
                    prob.boundary_conditions.terminal(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, guess.k))

    elif isinstance(prob, CompDualOCP):
        ocp_bc = prob.comp_ocp.boundary_conditions
        dual_bc = prob.comp_dual.adjoined_boundary_conditions

        def bc0_func(guess):
            psi_0 = ocp_bc.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, guess.k)

            adj_bc0 = dual_bc.initial(
                    guess.t[0], guess.x[:, 0], guess.lam[:, 0], guess.u[:, 0], guess.p, guess.nu0, guess.k)

            return np.concatenate((np.asarray(psi_0), np.asarray(adj_bc0)))

        def bcf_func(guess):
            psi_f = ocp_bc.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, guess.k)

            adj_bcf = dual_bc.terminal(
                    guess.t[-1], guess.x[:, -1], guess.lam[:, -1], guess.u[:, -1], guess.p, guess.nuf, guess.k)

            return np.concatenate((np.asarray(psi_f), np.asarray(adj_bcf)))

    elif isinstance(prob, AdiffBVP):
        def bc0_func(guess):
            return np.asarray(
                    prob.ca_boundary_conditions.initial(guess.t[0], guess.x[:, 0], guess.p, guess.k)).flatten()

        def bcf_func(guess):
            return np.asarray(
                    prob.ca_boundary_conditions.terminal(guess.t[-1], guess.x[:, -1], guess.p, guess.k)).flatten()

    elif isinstance(prob, AdiffOCP):
        def bc0_func(guess):
            return np.asarray(
                    prob.ca_boundary_conditions.initial(
                            guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, guess.k)).flatten()

        def bcf_func(guess):
            return np.asarray(
                    prob.ca_boundary_conditions.terminal(
                            guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, guess.k)).flatten()

    elif isinstance(prob, AdiffDualOCP):
        ocp_bc = prob.ocp.ca_boundary_conditions
        ocp_bc = prob.ocp.ca_boundary_conditions
        dual_bc = prob.dual.ca_adj_boundary_conditions

        def bc0_func(guess):
            psi_0 = ocp_bc.initial(guess.t[0], guess.x[:, 0], guess.u[:, 0], guess.p, guess.k)
            adj_bc0 = dual_bc.initial(
                    guess.t[0], guess.x[:, 0], guess.lam[:, 0], guess.u[:, 0], guess.p, guess.nu0, guess.k)

            return np.concatenate((np.asarray(psi_0).flatten(), np.asarray(adj_bc0).flatten()))

        def bcf_func(guess):
            psi_f = ocp_bc.terminal(guess.t[-1], guess.x[:, -1], guess.u[:, -1], guess.p, guess.k)
            adj_bcf = dual_bc.terminal(
                    guess.t[-1], guess.x[:, -1], guess.lam[:, -1], guess.u[:, -1], guess.p, guess.nuf, guess.k)

            return np.concatenate((np.asarray(psi_f).flatten(), np.asarray(adj_bcf).flatten()))

    else:
        raise ValueError(f'Problem type {type(prob)} not supported')

    return bc0_func, bcf_func
