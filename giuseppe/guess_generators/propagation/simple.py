from typing import Union, Optional, Callable, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, BVPSol, OCPSol, DualOCPSol
from giuseppe.problems.dual.utils import sift_ocp_and_dual
from ..constant import update_constant_value, generate_constant_guess

_ivp_sol = TypeVar('_ivp_sol')

SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP]
SUPPORTED_SOLUTIONS = Union[BVPSol, OCPSol, DualOCPSol]


# TODO Allow for reverse integration
def propagate_guess(comp_prob: SUPPORTED_PROBLEMS, default_value: float = 0.1, t_span: Union[float, ArrayLike] = 0.1,
                    initial_states: Optional[ArrayLike] = None, control: Optional[Union[float, ArrayLike]] = None,
                    p: Optional[Union[float, ArrayLike]] = None, k: Optional[Union[float, ArrayLike]] = None,
                    abs_tol: float = 1e-3, rel_tol: float = 1e-3) -> SUPPORTED_SOLUTIONS:

    prob, dual = sift_ocp_and_dual(comp_prob)
    guess = generate_constant_guess(comp_prob, default_value=default_value, t_span=t_span, x=initial_states, p=p, k=k)

    dynamics = prob.dynamics

    if initial_states is None:
        initial_states = guess.x[:, 0]

    if isinstance(prob, CompBVP):

        def wrapped_dynamics(t, x):
            return dynamics(t, x, guess.p, guess.k)

    elif isinstance(prob, CompOCP):
        if isinstance(control, Callable):
            def wrapped_dynamics(t, x):
                return dynamics(t, x, control(t, x, guess.p, guess.k), guess.p, guess.k)

        else:
            update_constant_value(guess, 'u', control)

            def wrapped_dynamics(t, x):
                return dynamics(t, x, guess.u[:, 0], guess.p, guess.k)

    else:
        raise ValueError(f'Problem type {type(comp_prob)} not supported')

    sol: _ivp_sol = solve_ivp(wrapped_dynamics, (guess.t[0], guess.t[-1]), initial_states, rtol=rel_tol, atol=abs_tol)

    guess.t = sol.t
    guess.x = sol.y

    if isinstance(prob, CompOCP):
        if isinstance(control, Callable):
            guess.u = np.array([control(ti, xi, guess.p, guess.k) for ti, xi in zip(guess.t, guess.x.T)]).T
        else:
            guess.u = np.vstack([guess.u[:, 0] for _ in guess.t]).T

    return guess
