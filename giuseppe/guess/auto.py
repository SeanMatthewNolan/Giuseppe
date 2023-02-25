from typing import Union, Optional, Callable

from numpy.typing import ArrayLike

from giuseppe.io import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual
from giuseppe.guess.initialize_guess import initialize_guess
from giuseppe.guess.sequential_linear_projection import match_constants_to_boundary_conditions, \
    match_states_to_boundary_conditions, match_states, match_adjoints

CONTROL_FUNC = Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike]


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
        fit_state_dynamics: bool = False,
        fit_adjoints: bool = True,
        fit_constants: bool = True,
        quadrature: str = 'linear'
) -> Solution:

    if problem.prob_class in ['bvp', 'ocp']:
        guess = initialize_guess(problem, default_value=default_value, t_span=t_span, x=x, u=u, p=p, k=k)

        guess = match_states_to_boundary_conditions(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol)

        if fit_state_dynamics:
            guess = match_states(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol, quadrature=quadrature)

    elif problem.prob_class == 'dual':
        guess = initialize_guess(problem, default_value=default_value, t_span=t_span, x=x, lam=lam, u=u,
                                 p=p, nu0=nu0, nuf=nuf, k=k)

        guess = match_states_to_boundary_conditions(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol)

        if fit_state_dynamics:
            guess = match_states(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol, quadrature=quadrature)

        if fit_adjoints:
            guess = match_adjoints(problem, guess, quadrature=quadrature, rel_tol=rel_tol, abs_tol=abs_tol)

    else:
        raise RuntimeError(f'Cannot process problem of class {type(problem)}')

    if fit_constants:
        guess = match_constants_to_boundary_conditions(problem, guess, rel_tol=rel_tol, abs_tol=abs_tol)

    return guess
