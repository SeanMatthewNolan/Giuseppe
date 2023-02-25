from typing import Union, Optional, Callable, TypeVar
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from giuseppe.io import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual
from giuseppe.guess_generation.initialize_guess import initialize_guess, process_static_value, process_dynamic_value

_IVP_SOL = TypeVar('_IVP_SOL')
CONTROL_FUNC = Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike]


def propagate_guess(
        problem: Union[BVP, OCP, Dual],
        t_span: Union[float, ArrayLike] = 1,
        reverse: bool = False,
        initial_states: Optional[ArrayLike] = None,
        initial_costates: Optional[ArrayLike] = None,
        control: Optional[Union[float, ArrayLike, CONTROL_FUNC]] = None,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        p: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None,
        nu0: Optional[ArrayLike] = None,
        nuf: Optional[ArrayLike] = None,
        default_value: float = 1.
) -> Solution:

    """
    Propagate a guess with a constant control input_value or control function.

    After propagation, projection may be used to estimate the dual variable (costates and adjoints) and match the
    constants to the guess.

    Unspecified values will be set to default_value.

    Parameters
    ----------
    problem : BVP, OCP, or Dual
        the problem that the guess is for
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
    abs_tol : float, default=1e-4
       absolute tolerance for propagation
    rel_tol : float, default=1e-4
       relative tolerance for propagation
    default_value : float, default=1
        input_value used if no input_value is given

    Returns
    -------
    guess

    """
    if problem.prob_class == 'bvp':
        guess = propagate_bvp_guess(
                problem, t_span, initial_states,
                p=p, k=k, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse, default_value=default_value)
    elif problem.prob_class == 'ocp':
        guess = propagate_ocp_guess(
                problem, t_span, initial_states, control,
                p=p,  k=k, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse, default_value=default_value)
    elif problem.prob_class == 'dual':
        guess = propagate_dual_guess(
                problem, t_span, initial_states, initial_costates, control,
                p=p, nu0=nu0, nuf=nuf, k=k,
                abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse, default_value=default_value)
    else:
        raise RuntimeError(f'Cannot process problem of class {type(problem)}')

    return guess


def propagate_bvp_guess(
        bvp: BVP,
        t_span: Union[float, ArrayLike],
        initial_states: Optional[ArrayLike],
        p: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None,
        default_value: float = 1.,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        reverse: bool = False,
) -> Solution:

    guess = initialize_guess(
            bvp, default_value=default_value, t_span=t_span, x=initial_states, p=p, k=k
    )

    return propagate_bvp_guess_from_guess(bvp, guess, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse)


def propagate_bvp_guess_from_guess(
        bvp: BVP,
        input_guess: Solution,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        reverse: bool = False,
) -> Solution:
    guess = deepcopy(input_guess)

    if reverse:
        guess.t = np.flip(guess.t)
        initial_states = guess.x[:, -1]
    else:
        initial_states = guess.x[:, 0]

    ivp_sol: _IVP_SOL = solve_ivp(
            lambda _t, _x: bvp.compute_dynamics(_t, _x, guess.p, guess.k),
            (guess.t[0], guess.t[-1]), initial_states, atol=abs_tol, rtol=rel_tol)
    guess.t = ivp_sol.t
    guess.x = ivp_sol.y

    if reverse:
        guess.t = np.flip(guess.t)
        guess.x = np.flip(guess.x, axis=1)

    return guess


def propagate_ocp_guess(
        ocp: OCP,
        t_span: Union[float, ArrayLike],
        initial_states: ArrayLike,
        control: Union[float, ArrayLike, Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike], None],
        p: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None,
        default_value: float = 1.,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        reverse: bool = False,
) -> Solution:

    guess = initialize_guess(ocp, default_value=default_value, t_span=t_span, x=initial_states, p=p, k=k)

    return propagate_ocp_guess_from_guess(ocp, guess, control, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse)


def propagate_ocp_guess_from_guess(
        ocp: OCP,
        input_guess: Solution,
        control: Union[float, ArrayLike, Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike], None],
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        reverse: bool = False,
) -> Solution:

    guess = deepcopy(input_guess)

    _compute_dynamics = ocp.compute_dynamics
    p, k = guess.p, guess.k

    if reverse:
        guess.t = np.flip(guess.t)
        initial_states = guess.x[:, -1]
    else:
        initial_states = guess.x[:, 0]

    if isinstance(control, Callable):
        def _compute_dynamics_wrapped(_t, _x):
            _u = np.asarray(control(_t, _x, p, k))
            return _compute_dynamics(_t, _x, _u, p, k)

    else:
        if control is None:
            control = guess.u[:, 0]

        u = process_static_value(control, ocp.num_controls)

        def _compute_dynamics_wrapped(_t, _x):
            return _compute_dynamics(_t, _x, u, p, k)

    ivp_sol: _IVP_SOL = solve_ivp(_compute_dynamics_wrapped, (guess.t[0], guess.t[-1]), initial_states,
                                  atol=abs_tol, rtol=rel_tol)
    guess.t = ivp_sol.t
    guess.x = ivp_sol.y

    if reverse:
        guess.t = np.flip(guess.t)
        guess.x = np.flip(guess.x, axis=1)

    if isinstance(control, Callable):
        guess.u = np.asarray([control(t_i, x_i, p, k) for t_i, x_i in zip(guess.t, guess.x.T)]).T
    else:
        guess.u = process_dynamic_value(control, (ocp.num_controls, len(guess.t)))

    return guess


def propagate_dual_guess(
        dual: Dual,
        t_span: Union[float, ArrayLike],
        initial_states: ArrayLike,
        initial_costates: ArrayLike,
        control: Union[float, ArrayLike, Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike], None],
        p: Optional[Union[float, ArrayLike]] = None,
        nu0: Optional[Union[float, ArrayLike]] = None,
        nuf: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None,
        default_value: float = 1.,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        reverse: bool = False,
) -> Solution:

    guess = initialize_guess(dual, default_value=default_value, t_span=t_span, x=initial_states, lam=initial_costates,
                             p=p, nu0=nu0, nuf=nuf, k=k)

    return propagate_dual_guess_from_guess(dual, guess, control, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse)


def propagate_dual_guess_from_guess(
        dual: Dual,
        input_guess: Solution,
        control: Union[float, ArrayLike, Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike], None],
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        reverse: bool = False,
) -> Solution:

    guess = deepcopy(input_guess)
    num_states, num_costates, _compute_dynamics, _compute_costate_dynamics = dual.num_states, dual.num_costates,\
        dual.compute_dynamics, dual.compute_costate_dynamics
    p, nu0, nuf, k = guess.p, guess.nu0, guess.nuf, guess.k

    if reverse:
        guess.t = np.flip(guess.t)
        initial_states = guess.x[:, -1]
        initial_costates = guess.lam[:, -1]
    else:
        initial_states = guess.x[:, 0]
        initial_costates = guess.lam[:, 0]

    if isinstance(control, Callable):
        def _compute_dynamics_wrapped(_t, _y):
            _x, _lam = _y[:num_states], _y[num_states:num_states + num_costates]
            _u = np.asarray(control(_t, _x, p, k))
            _x_dot = _compute_dynamics(_t, _x, _u, p, k)
            _lam_dot = _compute_costate_dynamics(_t, _x, _lam, _u, p, k)
            return np.concatenate((_x_dot, _lam_dot))

    else:
        if control is None:
            control = guess.u[:, 0]

        u = process_static_value(control, dual.num_controls)

        def _compute_dynamics_wrapped(_t, _y):
            _x, _lam = _y[:num_states], _y[num_states:num_states + num_costates]
            _x_dot = _compute_dynamics(_t, _x, u, p, k)
            _lam_dot = _compute_costate_dynamics(_t, _x, _lam, u, p, k)
            return np.concatenate((_x_dot, _lam_dot))

    ivp_sol: _IVP_SOL = solve_ivp(_compute_dynamics_wrapped, (guess.t[0], guess.t[-1]),
                                  np.concatenate((initial_states, initial_costates)),
                                  atol=abs_tol, rtol=rel_tol)
    guess.t = ivp_sol.t
    guess.x = ivp_sol.y[:num_states, :]
    guess.lam = ivp_sol.y[num_states:num_states + num_costates, :]

    if reverse:
        guess.t = np.flip(guess.t)
        guess.x = np.flip(guess.x, axis=1)
        guess.lam = np.flip(guess.lam, axis=1)

    if isinstance(control, Callable):
        guess.u = np.asarray([control(t_i, x_i, p, k) for t_i, x_i in zip(guess.t, guess.x.T)]).T
    else:
        guess.u = process_dynamic_value(control, (dual.num_controls, len(guess.t)))

    return guess
