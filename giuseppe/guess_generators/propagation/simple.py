from typing import Union, Optional, Callable, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from giuseppe.utils.conversion import ca_vec2arr
from giuseppe.io import Solution
from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP
from ..constant import update_constant_value, initialize_guess_for_auto
from ..projection import match_constants_to_bcs, project_dual

_IVP_SOL = TypeVar('_IVP_SOL')
CONTROL_FUNC = Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike]
SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP]


# TODO Allow for reverse integration
def propagate_guess(
        comp_prob: SUPPORTED_PROBLEMS, default: Union[float, Solution] = 0.1,
        t_span: Union[float, ArrayLike] = 0.1, initial_states: Optional[ArrayLike] = None,
        initial_costates: Optional[ArrayLike] = None, control: Optional[Union[float, ArrayLike, CONTROL_FUNC]] = None,
        p: Optional[Union[float, ArrayLike]] = None, k: Optional[Union[float, ArrayLike]] = None,
        use_project_dual: bool = True, use_match_constants: bool = True, reverse: bool = False,
        abs_tol: float = 1e-3, rel_tol: float = 1e-3) -> Solution:
    """
    Propagate a guess with a constant control value or control function.

    After propagation, projection may be used to estimate the dual variable (costates and adjoints) and match the
    constants to the guess.

    Unspecified values will be set to default.

    Parameters
    ----------
    comp_prob : CompBVP, CompOCP, or CompDualOCP
        the problem that the guess is for
    default : float, BVPSol, OCPSol or DualOCPSol, default=0.1
        value used if no value is given
    t_span : float or ArrayLike, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])
    initial_states : ArrayLike, optional
        state values to start propagation
    initial_costates : ArrayLike, optional
        costate values to start propagation, might change with use_project_dual=True
    control : float, Arraylike, or Callable[[float, ArrayLike, ArrayLike, Arraylike], ArrayLike]
        control used in propagation
        if float, all control values are set to that value
        if ArrayLike, control is set to constant specified array
        if Callable, callable function is called to compute control at each time
    p : ArrayLike, optional
        parameter values
    k : ArrayLike, optional
        constant values
        updated at end if use_match_constants=True
    use_match_constants : bool, default=True
        if True, match_constants will be called to update the constants to most closely match the formed guess
    use_project_dual : bool, default=True
        if True and comp_prob is an instance CompDualOCP, project_dual will be called to estimate dual values from guess
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

    guess = initialize_guess_for_auto(comp_prob, default=default, t_span=t_span, constants=k)

    if p is not None:
        update_constant_value(guess, 'p', p)

    if initial_states is None:
        if not reverse:
            initial_states = guess.x[:, 0]
        else:
            initial_states = guess.x[:, -1]

    if not reverse:
        prop_t_span = (guess.t[0], guess.t[-1])
    else:
        prop_t_span = (guess.t[-1], guess.t[0])

    if isinstance(comp_prob, CompBVP):
        dynamics = comp_prob.dynamics

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

    elif isinstance(comp_prob, CompOCP):
        dynamics = comp_prob.dynamics

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

    elif isinstance(comp_prob, AdiffBVP):
        dynamics = comp_prob.ca_dynamics

        def wrapped_dynamics(t, x):
            return ca_vec2arr(dynamics(t, x, guess.p, guess.k))

        sol: _IVP_SOL = solve_ivp(
                wrapped_dynamics, (guess.t[0], guess.t[-1]), initial_states, rtol=rel_tol, atol=abs_tol)

        guess.t = sol.t
        guess.x = sol.y

    elif isinstance(comp_prob, AdiffOCP):
        dynamics = comp_prob.ca_dynamics

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

    elif isinstance(comp_prob, AdiffDualOCP):
        if initial_costates is None:
            initial_costates = guess.lam[:, 0]

        dynamics = comp_prob.ocp.ca_dynamics
        costate_dynamics = comp_prob.dual.ca_costate_dynamics

        num_x = comp_prob.ocp.num_states
        num_lam = comp_prob.dual.num_costates
        num_control = comp_prob.dual.num_controls

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
        raise ValueError(f'Problem type {type(comp_prob)} not supported')

    if isinstance(comp_prob, CompOCP) or isinstance(comp_prob, CompDualOCP)\
            or isinstance(comp_prob, AdiffOCP) or isinstance(comp_prob, AdiffDualOCP):
        if isinstance(control, Callable):
            guess.u = np.array([control(ti, xi, guess.p, guess.k) for ti, xi in zip(guess.t, guess.x.T)]).T
        else:
            guess.u = np.vstack([guess.u[:, 0] for _ in guess.t]).T

    if (isinstance(comp_prob, CompDualOCP) or isinstance(comp_prob, AdiffDualOCP)) and use_project_dual:
        guess = project_dual(comp_prob, guess, rel_tol=rel_tol, abs_tol=abs_tol)

    if use_match_constants:
        guess = match_constants_to_bcs(comp_prob, guess, rel_tol=rel_tol, abs_tol=abs_tol)

    return guess
