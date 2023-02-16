from typing import Union, Optional, Callable, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from giuseppe.io import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual

_IVP_SOL = TypeVar('_IVP_SOL')
CONTROL_FUNC = Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike]
SUPPORTED_PROBLEMS = Union[BVP, OCP, Dual]


# TODO Allow for reverse integration
def propagate_guess(
        problem: SUPPORTED_PROBLEMS, t_span: Union[float, ArrayLike] = 0.1,
        initial_states: Optional[ArrayLike] = None, initial_costates: Optional[ArrayLike] = None,
        control: Optional[Union[float, ArrayLike, CONTROL_FUNC]] = None,
        p: Optional[Union[float, ArrayLike]] = None, k: Optional[Union[float, ArrayLike]] = None,
        nu0: Optional[ArrayLike] = None, nuf: Optional[ArrayLike] = None, reverse: bool = False,
        abs_tol: float = 1e-3, rel_tol: float = 1e-3,
) -> Solution:

    """
    Propagate a guess with a constant control input_value or control function.

    After propagation, projection may be used to estimate the dual variable (costates and adjoints) and match the
    constants to the guess.

    Unspecified values will be set to default.

    Parameters
    ----------
    problem : CompBVP, CompOCP, or CompDualOCP
        the problem that the guess is for
    default : float, BVPSol, OCPSol or DualOCPSol, default=0.1
        input_value used if no input_value is given
    t_span : float or ArrayLike, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])
    initial_states : ArrayLike, optional
        state values to start propagation
    initial_costates : ArrayLike, optional
        costate values to start propagation, might change with use_project_dual=True
    control : float, Arraylike, or Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike]
        control used in propagation
        if float, all control values are set to that input_value
        if ArrayLike, control is set to constant specified array
        if Callable, callable function is called to compute control at each time u(t, x, p, k)
    p : ArrayLike, optional
        parameter values
    k : ArrayLike, optional
        constant values
        updated at end if use_match_constants=True
    nu0 : ArrayLike, optional
        initial adjoints
    nuf : ArrayLike, optional
        terminal adjoints
    reverse : bool, default=False
        if True, will propagate solution from the terminal location
    abs_tol : float, default=1e-3
       absolute tolerance for propagation
    rel_tol : float, default=1e-3
       relative tolerance for propagation

    Returns
    -------
    guess

    """

    if isinstance(t_span, float) or isinstance(t_span, int):
        t = np.asarray([0., t_span], dtype=float)
    else:
        t = np.asarray(t_span, dtype=float)


    if p is not None:
        update_constant_value(guess, 'p', p)

    if nu0 is not None:
        update_constant_value(guess, 'nu0', nu0)

    if nuf is not None:
        update_constant_value(guess, 'nuf', nuf)

    if initial_states is None:
        if not reverse:
            initial_states = guess.x[:, 0]
        else:
            initial_states = guess.x[:, -1]

    if not reverse:
        prop_t_span = (guess.t[0], guess.t[-1])
    else:
        prop_t_span = (guess.t[-1], guess.t[0])

    if isinstance(problem, CompBVP):
        dynamics = problem.dynamics

        def wrapped_dynamics(t, x):
            return dynamics(t, x, guess.p, guess.k)

        sol: _IVP_SOL = solve_ivp(
                wrapped_dynamics, prop_t_span, initial_states, rtol=rel_tol, atol=abs_tol)

        if not reverse:
            guess.t = sol.t
            guess.x = sol.y
        else:
            guess.t = sol.t[::-1]
            guess.x = sol.y[:, ::-1]

    elif isinstance(problem, CompOCP):
        dynamics = problem.dynamics

        if isinstance(control, Callable):
            def wrapped_dynamics(t, x):
                return dynamics(t, x, control(t, x, guess.p, guess.k), guess.p, guess.k)

        else:
            update_constant_value(guess, 'u', control)

            def wrapped_dynamics(t, x):
                return dynamics(t, x, guess.u[:, 0], guess.p, guess.k)

        sol: _IVP_SOL = solve_ivp(wrapped_dynamics, prop_t_span, initial_states, rtol=rel_tol, atol=abs_tol)

        if not reverse:
            guess.t = sol.t
            guess.x = sol.y
        else:
            guess.t = sol.t[::-1]
            guess.x = sol.y[:, ::-1]

    elif isinstance(problem, CompDualOCP):
        if initial_costates is None:
            initial_costates = guess.lam[:, 0]

        dynamics = problem.comp_ocp.dynamics
        costate_dynamics = problem.comp_dual.costate_dynamics

        num_x = problem.comp_ocp.num_states
        num_lam = problem.comp_dual.num_costates

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

        sol: _IVP_SOL = solve_ivp(
                wrapped_dynamics, prop_t_span, np.concatenate((initial_states, initial_costates)),
                rtol=rel_tol, atol=abs_tol)

        if not reverse:
            guess.t = sol.t
            guess.x = sol.y[:num_x]
            guess.lam = sol.y[num_x:num_x + num_lam]
        else:
            guess.t = sol.t[::-1]
            guess.x = sol.y[:num_x, ::-1]
            guess.lam = sol.y[num_x:num_x + num_lam, ::-1]

    elif isinstance(problem, AdiffBVP):
        dynamics = problem.ca_dynamics

        def wrapped_dynamics(t, x):
            return ca_vec2arr(dynamics(t, x, guess.p, guess.k))

        sol: _IVP_SOL = solve_ivp(
                wrapped_dynamics, (guess.t[0], guess.t[-1]), initial_states, rtol=rel_tol, atol=abs_tol)

        guess.t = sol.t
        guess.x = sol.y

    elif isinstance(problem, AdiffOCP):
        dynamics = problem.ca_dynamics

        if isinstance(control, Callable):
            def wrapped_dynamics(t, x):
                return ca_vec2arr(dynamics(t, x, control(t, x, guess.p, guess.k), guess.p, guess.k))

        else:
            update_constant_value(guess, 'u', control)

            def wrapped_dynamics(t, x):
                return ca_vec2arr(dynamics(t, x, guess.u[:, 0], guess.p, guess.k))

        sol: _IVP_SOL = solve_ivp(
                wrapped_dynamics, (guess.t[0], guess.t[-1]), initial_states, rtol=rel_tol, atol=abs_tol)

        guess.t = sol.t
        guess.x = sol.y

    elif isinstance(problem, AdiffDualOCP):
        if initial_costates is None:
            initial_costates = guess.lam[:, 0]

        dynamics = problem.ocp.ca_dynamics
        costate_dynamics = problem.dual.ca_costate_dynamics

        num_x = problem.ocp.num_states
        num_lam = problem.dual.num_costates
        num_control = problem.dual.num_controls

        if isinstance(control, Callable):
            def wrapped_dynamics(t, y):
                x = y[:num_x]
                lam = y[num_x:num_x + num_lam]
                u = control(t, x, guess.p, guess.k)

                x_dot = dynamics(t, x, u, guess.p, guess.k)
                lam_dot = costate_dynamics(t, x, lam, u, guess.p, guess.k)

                return np.concatenate((ca_vec2arr(x_dot), ca_vec2arr(lam_dot)))

        else:
            update_constant_value(guess, 'u', control)

            def wrapped_dynamics(t, y):
                x = y[:num_x]
                lam = y[num_x:num_x + num_lam]
                u = guess.u[:, 0]

                x_dot = dynamics(t, x, u, guess.p, guess.k)
                lam_dot = costate_dynamics(t, x, lam, u, guess.p, guess.k)

                return np.concatenate((ca_vec2arr(x_dot), ca_vec2arr(lam_dot)))

        sol: _IVP_SOL = solve_ivp(
                wrapped_dynamics, (guess.t[0], guess.t[-1]), np.concatenate((initial_states, initial_costates)),
                rtol=rel_tol, atol=abs_tol)

        guess.t = sol.t
        guess.x = sol.y[:num_x]
        guess.lam = sol.y[num_x:num_x + num_lam]

    else:
        raise ValueError(f'Problem type {type(problem)} not supported')

    return guess
