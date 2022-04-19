from typing import Union, Optional, Callable, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, BVPSol, OCPSol, DualOCPSol
from ..constant import update_constant_value, generate_constant_guess
from ..projection import match_constants_to_bcs, project_dual

_ivp_sol = TypeVar('_ivp_sol')

SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP]
SUPPORTED_SOLUTIONS = Union[BVPSol, OCPSol, DualOCPSol]


# TODO Allow for reverse integration
def propagate_guess(comp_prob: SUPPORTED_PROBLEMS, default_value: float = 0.1, t_span: Union[float, ArrayLike] = 0.1,
                    initial_states: Optional[ArrayLike] = None, initial_costates: Optional[ArrayLike] = None,
                    control: Optional[Union[float, ArrayLike]] = None,
                    p: Optional[Union[float, ArrayLike]] = None, k: Optional[Union[float, ArrayLike]] = None,
                    use_match_constants: bool = True, use_project_dual: bool = True,
                    abs_tol: float = 1e-3, rel_tol: float = 1e-3) -> SUPPORTED_SOLUTIONS:

    guess = generate_constant_guess(comp_prob, default_value=default_value, t_span=t_span, x=initial_states, p=p, k=k)

    if initial_states is None:
        initial_states = guess.x[:, 0]

    if isinstance(comp_prob, CompBVP):
        dynamics = comp_prob.dynamics

        def wrapped_dynamics(t, x):
            return dynamics(t, x, guess.p, guess.k)

        sol: _ivp_sol = solve_ivp(
                wrapped_dynamics, (guess.t[0], guess.t[-1]), initial_states, rtol=rel_tol, atol=abs_tol)

        guess.t = sol.t
        guess.x = sol.y

    elif isinstance(comp_prob, CompOCP):
        dynamics = comp_prob.dynamics

        if isinstance(control, Callable):
            def wrapped_dynamics(t, x):
                return dynamics(t, x, control(t, x, guess.p, guess.k), guess.p, guess.k)

        else:
            update_constant_value(guess, 'u', control)

            def wrapped_dynamics(t, x):
                return dynamics(t, x, guess.u[:, 0], guess.p, guess.k)

        sol: _ivp_sol = solve_ivp(wrapped_dynamics, (guess.t[0], guess.t[-1]), initial_states, rtol=rel_tol, atol=abs_tol)

        guess.t = sol.t
        guess.x = sol.y

    elif isinstance(comp_prob, CompDualOCP):
        if initial_costates is None:
            initial_costates = guess.lam[:, 0]

        dynamics = comp_prob.comp_ocp.dynamics
        costate_dynamics = comp_prob.comp_dual.costate_dynamics

        num_x = comp_prob.comp_ocp.num_states
        num_lam = comp_prob.comp_dual.num_costates

        if isinstance(control, Callable):
            def wrapped_dynamics(t, y):
                x = y[:num_x]
                lam = y[num_x:num_x + num_lam]
                u = control(t, x, guess.p, guess.k)

                x_dot = dynamics(t, x, u, guess.p, guess.k)
                lam_dot = costate_dynamics(t, x, lam, u, guess.p, guess.k)

                return np.concatenate((x_dot, lam_dot))

        else:
            update_constant_value(guess, 'u', control)

            def wrapped_dynamics(t, y):
                x = y[:num_x]
                lam = y[num_x:num_x + num_lam]
                u = guess.u[:, 0]

                x_dot = dynamics(t, x, u, guess.p, guess.k)
                lam_dot = costate_dynamics(t, x, lam, u, guess.p, guess.k)

                return np.concatenate((x_dot, lam_dot))

        sol: _ivp_sol = solve_ivp(
                wrapped_dynamics, (guess.t[0], guess.t[-1]), np.concatenate((initial_states, initial_costates)),
                rtol=rel_tol, atol=abs_tol)

        guess.t = sol.t
        guess.x = sol.y[:num_x]
        guess.lam = sol.y[num_x:num_x + num_lam]

    else:
        raise ValueError(f'Problem type {type(comp_prob)} not supported')

    if isinstance(comp_prob, CompOCP) or isinstance(comp_prob, CompDualOCP):
        if isinstance(control, Callable):
            guess.u = np.array([control(ti, xi, guess.p, guess.k) for ti, xi in zip(guess.t, guess.x.T)]).T
        else:
            guess.u = np.vstack([guess.u[:, 0] for _ in guess.t]).T

    if isinstance(comp_prob, CompDualOCP) and use_project_dual:
        guess = project_dual(comp_prob, guess)

    if use_match_constants:
        guess = match_constants_to_bcs(comp_prob, guess)

    return guess
