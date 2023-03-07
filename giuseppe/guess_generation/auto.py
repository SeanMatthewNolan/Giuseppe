from typing import Union, Optional, Callable

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual
from giuseppe.guess_generation.initialize_guess import initialize_guess
from giuseppe.guess_generation.gauss_newton import match_constants_to_boundary_conditions, \
    match_states_to_boundary_conditions, match_states, match_adjoints

CONTROL_FUNC = Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike]

DEFAULT_NUM_NODES = 3


def auto_guess(
        problem: Union[BVP, OCP, Dual],
        default_value: float = 1.,
        t_span: Union[float, ArrayLike] = 1.,
        x: Optional[Union[ArrayLike, float]] = None,
        p: Optional[Union[ArrayLike, float]] = None,
        u: Optional[Union[ArrayLike, float]] = None,
        k: Optional[Union[float, ArrayLike]] = None,
        lam: Optional[Union[ArrayLike, float]] = None,
        nu0: Optional[Union[ArrayLike, float]] = None,
        nuf: Optional[Union[ArrayLike, float]] = None,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        fit_state_dynamics: bool = True,
        fit_adjoints: bool = True,
        fit_constants: bool = False,
        quadrature: str = 'trapezoidal',
        verbose: bool = False
) -> Solution:

    if isinstance(t_span, (float, int)):
        t_span = np.linspace(0., t_span, DEFAULT_NUM_NODES)

    if problem.prob_class in ['bvp', 'ocp']:
        guess = initialize_guess(problem, default_value=default_value, t_span=t_span, x=x, u=u, p=p, k=k)

        if fit_state_dynamics:
            if verbose:
                print(f'Matching states, parameters, and time to dynamics:')

            guess = match_states(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol, quadrature=quadrature,
                                 verbose=verbose)

        else:
            if verbose:
                print(f'Matching states, parameters, and time to boundary conditions:')

            guess = match_states_to_boundary_conditions(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol,
                                                        verbose=verbose)

    elif problem.prob_class == 'dual':
        guess = initialize_guess(problem, default_value=default_value, t_span=t_span, x=x, lam=lam, u=u,
                                 p=p, nu0=nu0, nuf=nuf, k=k)

        if fit_state_dynamics:
            if verbose:
                print(f'Matching states, parameters, and time to dynamics:')

            guess = match_states(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol, quadrature=quadrature,
                                 verbose=verbose)

        else:
            if verbose:
                print(f'Matching states, parameters, and time to boundary conditions:')

            guess = match_states_to_boundary_conditions(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol,
                                                        verbose=verbose)

        if fit_adjoints:
            if verbose:
                print(f'Fitting the costates and adjoint parameters:')

            guess = match_adjoints(problem, guess, quadrature=quadrature, rel_tol=rel_tol, abs_tol=abs_tol,
                                   verbose=verbose)

    else:
        raise RuntimeError(f'Cannot process problem of class {type(problem)}')

    if fit_constants:
        if verbose:
            print(f'Matching the constants to the boundary conditions:')

        guess = match_constants_to_boundary_conditions(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol,
                                                       verbose=verbose)

    return guess
