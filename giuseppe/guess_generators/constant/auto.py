from copy import deepcopy
from typing import Union, Optional
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, BVPSol, OCPSol, DualOCPSol
from giuseppe.problems.dual.utils import sift_ocp_and_dual
from .simple import generate_single_constant_guess, update_value_constant
from ..projection import project_to_nullspace

SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP]
SUPPORTED_SOLUTIONS = Union[BVPSol, OCPSol, DualOCPSol]


def auto_constant_guess(comp_prob: SUPPORTED_PROBLEMS, t_span: Union[float, ArrayLike] = 0.1,
                        constants: Optional[ArrayLike] = None, default: Union[float, SUPPORTED_SOLUTIONS] = 0.1,
                        abs_tol: float = 1e-3, rel_tol: float = 1e-3) -> SUPPORTED_SOLUTIONS:

    """
    Automatically generate guess where variables (excluding the indenpendent) are set to constant values

    Parameters
    ----------
    comp_prob : CompBVP, CompOCP, or CompDualOCP 
    t_span : float or ArrayLike, default=0.1
    constants : ArrayLike, optional
    default : float, BVPSol, OCPSol, or DualOCPSol, default=0.1
        if float, default value for set variables
        if solution type, guess to modify from which unmodified values will be taken
    abs_tol : float, default=1e-3
        absolute tolerance
    rel_tol : float, default=1e-3
        relative tolerance

    Returns
    -------
    guess : BVPSol, OCPSol, or DualOCPSol

    """
    prob, dual = sift_ocp_and_dual(comp_prob)

    if isinstance(default, BVPSol):
        guess = deepcopy(default)
    else:
        guess = generate_single_constant_guess(comp_prob, constant=default, t_span=t_span)

    if constants is not None:
        if constants.shape != prob.default_values.shape:
            warn(f'Inconsistant constants shape! Expected {prob.default_values.shape}')
        guess.k = constants

    # Project states, (controls, ) and parameters to BCs
    num_x = prob.num_states
    num_p = prob.num_parameters

    if isinstance(prob, CompBVP):
        def map_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            _x = values[:num_x]
            _p = values[num_x:num_x + num_p]
            return _x, _p

        def bc_func(values):
            _x, _p = map_values(values)
            psi_0 = prob.boundary_conditions.initial(guess.t[0], _x, _p, guess.k)
            psi_f = prob.boundary_conditions.terminal(guess.t[-1], _x, _p, guess.k)
            return np.concatenate((np.asarray(psi_0), np.asarray(psi_f)))

        values_guess = np.concatenate((np.mean(guess.x, axis=1), guess.p))
        out_x, out_p = map_values(project_to_nullspace(bc_func, values_guess, abs_tol=abs_tol, rel_tol=rel_tol))
        update_value_constant(guess, 'x', out_x)
        update_value_constant(guess, 'p', out_p)

    elif isinstance(prob, CompOCP):

        num_u = prob.num_controls

        def map_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            _x = values[:num_x]
            _u = values[num_x:num_x + num_u]
            _p = values[num_x + num_u:num_x + num_u + num_p]
            return _x, _u, _p

        def bc_func(values):
            _x, _u, _p = map_values(values)
            psi_0 = prob.boundary_conditions.initial(guess.t[0], _x, _u, _p, guess.k)
            psi_f = prob.boundary_conditions.terminal(guess.t[-1], _x, _u, _p, guess.k)
            return np.concatenate((np.asarray(psi_0), np.asarray(psi_f)))

        values_guess = np.concatenate((np.mean(guess.x, axis=1), np.mean(guess.u, axis=1), guess.p))
        out_x, out_u, out_p = map_values(project_to_nullspace(bc_func, values_guess, abs_tol=abs_tol, rel_tol=rel_tol))
        update_value_constant(guess, 'x', out_x)
        update_value_constant(guess, 'u', out_u)
        update_value_constant(guess, 'p', out_p)

        if dual is not None:

            num_lam = dual.num_costates
            num_nu0 = dual.num_initial_adjoints
            num_nuf = dual.num_terminal_adjoints

            ocp_bc = prob.boundary_conditions
            dual_bc = dual.adjoined_boundary_conditions

            def map_dual_values(values: np.ndarray) \
                    -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

                _x = values[:num_x]
                _lam = values[num_x:num_x + num_lam]
                _u = values[num_x + num_lam:num_x + num_lam + num_u]
                _p = values[num_x + num_lam + num_u:num_x + num_lam + num_u + num_p]
                _nu0 = values[num_x + num_lam + num_u + num_p:num_x + num_lam + num_u + num_p + num_nu0]
                _nuf = values[num_x + num_lam + num_u + num_p + num_nu0:
                              num_x + num_lam + num_u + num_p + num_nu0 + num_nuf]
                return _x, _lam, _u, _p, _nu0, _nuf

            def dual_bc_func(values):
                _x, _lam, _u, _p, _nu0, _nuf = map_dual_values(values)

                psi_0 = ocp_bc.initial(guess.t[0], _x, _u, _p, guess.k)
                psi_f = ocp_bc.terminal(guess.t[-1], _x, _u, _p, guess.k)

                adj_bc0 = dual_bc.initial(guess.t[0], _x, _lam, _u, _p, _nu0, guess.k)
                adj_bcf = dual_bc.terminal(guess.t[-1], _x, _lam, _u, _p, _nuf, guess.k)

                return np.concatenate((np.asarray(psi_0), np.asarray(psi_f), np.asarray(adj_bc0), np.asarray(adj_bcf)))

            values_guess = np.concatenate(
                    (np.mean(guess.x, axis=1), np.mean(guess.lam, axis=1), np.mean(guess.u, axis=1),
                     guess.p, guess.nu0, guess.nuf)
            )
            out_x, out_lam, out_u, out_p, out_nu0, out_nuf = map_dual_values(
                    project_to_nullspace(dual_bc_func, values_guess, abs_tol=abs_tol, rel_tol=rel_tol)
            )
            update_value_constant(guess, 'x', out_x)
            update_value_constant(guess, 'lam', out_lam)
            update_value_constant(guess, 'u', out_u)
            update_value_constant(guess, 'p', out_p)
            update_value_constant(guess, 'nu0', out_nu0)
            update_value_constant(guess, 'nuf', out_nuf)

    else:
        raise ValueError(f'Problem type {type(comp_prob)} not supported')

    return guess
