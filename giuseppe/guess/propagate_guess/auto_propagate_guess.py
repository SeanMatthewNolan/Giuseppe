from typing import Union, Optional, Callable

from numpy.typing import ArrayLike

from giuseppe.io import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual
from giuseppe.guess.initialize_guess import initialize_guess, process_dynamic_value
from giuseppe.guess.propagate_guess import propagate_bvp_guess_from_guess, propagate_ocp_guess_from_guess,\
    propagate_dual_guess_from_guess
from giuseppe.guess.sequential_linear_projection import match_constants_to_boundary_conditions, match_states_to_boundary_conditions,\
    match_adjoints

CONTROL_FUNC = Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike]


def auto_propagate_guess(
        problem: Union[BVP, OCP, Dual],
        t_span: Union[float, ArrayLike] = 1,
        reverse: bool = False,
        initial_states: Optional[ArrayLike] = None,
        initial_costates: Optional[ArrayLike] = None,
        control: Union[float, ArrayLike, CONTROL_FUNC, None] = None,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        p: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None,
        nu0: Optional[ArrayLike] = None,
        nuf: Optional[ArrayLike] = None,
        default_value: float = 1.,
        match_constants: bool = True,
) -> Solution:

    """
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
        guess = auto_propagate_bvp_guess(
                problem, t_span, initial_states,
                p=p, k=k, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse, default_value=default_value,
                match_constants=match_constants)
    elif problem.prob_class == 'ocp':
        guess = auto_propagate_ocp_guess(
                problem, t_span, initial_states, control,
                p=p,  k=k, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse, default_value=default_value)
    elif problem.prob_class == 'dual':
        guess = auto_propagate_dual_guess(
                problem, t_span, initial_states, initial_costates, control,
                p=p, nu0=nu0, nuf=nuf, k=k,
                abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse, default_value=default_value)
    else:
        raise RuntimeError(f'Cannot process problem of class {type(problem)}')

    return guess


def auto_propagate_bvp_guess(
        bvp: BVP,
        t_span: Union[float, ArrayLike],
        initial_states: Optional[ArrayLike] = None,
        p: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None,
        default_value: float = 1.,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        reverse: bool = False,
        match_constants: bool = True
) -> Solution:
    guess = initialize_guess(bvp, default_value=default_value, t_span=t_span, x=initial_states, p=p, k=k)
    guess = match_states_to_boundary_conditions(bvp, guess, rel_tol=rel_tol, abs_tol=abs_tol)
    guess = propagate_bvp_guess_from_guess(bvp, guess, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse)
    if match_constants:
        guess = match_constants_to_boundary_conditions(bvp, guess, rel_tol=rel_tol, abs_tol=abs_tol)

    return guess


def auto_propagate_ocp_guess(
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
        match_constants: bool = True
) -> Solution:
    guess = initialize_guess(ocp, default_value=default_value, t_span=t_span, x=initial_states, p=p, k=k)
    guess = match_states_to_boundary_conditions(ocp, guess, rel_tol=rel_tol, abs_tol=abs_tol)
    guess = propagate_ocp_guess_from_guess(
            ocp, guess, control=control, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse)
    if match_constants:
        guess = match_constants_to_boundary_conditions(ocp, guess, rel_tol=rel_tol, abs_tol=abs_tol)

    return guess


def auto_propagate_dual_guess(
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
        match_constants: bool = True,
        fit_adjoints: bool = True,
        quadrature: str = 'linear',
) -> Solution:
    guess = initialize_guess(dual, default_value=default_value, t_span=t_span, x=initial_states, lam=initial_costates,
                             p=p, nu0=nu0, nuf=nuf, k=k)
    guess = match_states_to_boundary_conditions(dual, guess, rel_tol=rel_tol, abs_tol=abs_tol)

    if fit_adjoints:
        guess = propagate_ocp_guess_from_guess(
                dual, guess, control=control, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse)
        guess.lam = process_dynamic_value(initial_costates, guess.x.shape)
        guess = match_adjoints(dual, guess, quadrature, rel_tol=rel_tol, abs_tol=abs_tol)
    else:
        guess = propagate_dual_guess_from_guess(
                dual, guess, control=control, abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse)

    if match_constants:
        guess = match_constants_to_boundary_conditions(dual, guess, rel_tol=rel_tol, abs_tol=abs_tol)

    return guess
