from typing import Union, Optional, Callable

from numpy.typing import ArrayLike

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import BVP, OCP, Dual
from giuseppe.guess_generation.initialize_guess import initialize_guess, process_dynamic_value
from giuseppe.guess_generation.propagate_guess import propagate_bvp_guess_from_guess, propagate_ocp_guess_from_guess,\
    propagate_dual_guess_from_guess
from giuseppe.guess_generation.gauss_newton import match_constants_to_boundary_conditions,\
    match_states_to_boundary_conditions, match_adjoints

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
        max_step: Optional[float] = None,
        p: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None,
        nu0: Optional[ArrayLike] = None,
        nuf: Optional[ArrayLike] = None,
        default_value: float = 1.,
        match_constants: bool = True,
        fit_adjoints: bool = True,
        quadrature: str = 'trapezoidal',
        condition_adjoints: bool = False,
        verbose: bool = False
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
    max_step : float, optional
    default_value : float, default=1
        input_value used if no input_value is given
    match_constants: bool, default=True
    fit_adjoints: bool, default=True
    quadrature: str, default='simpson'
    verbose : bool, default=False

    Returns
    -------
    guess

    """
    if problem.prob_class == 'bvp':
        guess = auto_propagate_bvp_guess(
                problem, t_span, initial_states,
                p=p, k=k, abs_tol=abs_tol, rel_tol=rel_tol, max_step=max_step, reverse=reverse,
                default_value=default_value, match_constants=match_constants, verbose=verbose
        )
    elif problem.prob_class == 'ocp':
        guess = auto_propagate_ocp_guess(
                problem, t_span, initial_states, control,
                p=p,  k=k, abs_tol=abs_tol, rel_tol=rel_tol, max_step=max_step, reverse=reverse,
                match_constants=match_constants, default_value=default_value, verbose=verbose
        )
    elif problem.prob_class == 'dual':
        guess = auto_propagate_dual_guess(
                problem, t_span, initial_states, initial_costates, control,
                p=p, nu0=nu0, nuf=nuf, k=k,
                abs_tol=abs_tol, rel_tol=rel_tol, max_step=max_step, reverse=reverse, default_value=default_value,
                match_constants=match_constants, fit_adjoints=fit_adjoints, condition_adjoints=condition_adjoints,
                quadrature=quadrature, verbose=verbose
        )
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
        max_step: Optional[float] = None,
        reverse: bool = False,
        match_constants: bool = True,
        verbose: bool = False
) -> Solution:
    guess = initialize_guess(bvp, default_value=default_value, t_span=t_span, x=initial_states, p=p, k=k)

    if verbose:
        print(f'Matching states, parameters, and time to boundary conditions:')
    guess = match_states_to_boundary_conditions(bvp, guess, rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)

    if verbose:
        print(f'Propagating the dynamics in time\n')
    guess = propagate_bvp_guess_from_guess(bvp, guess,
        abs_tol=abs_tol, rel_tol=rel_tol, reverse=reverse, max_step=max_step)

    if match_constants:
        if verbose:
            print(f'Matching states, parameters, and time to dynamics:')
        guess = match_constants_to_boundary_conditions(bvp, guess, rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)

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
        max_step: Optional[float] = None,
        reverse: bool = False,
        match_constants: bool = True,
        verbose: bool = False
) -> Solution:
    guess = initialize_guess(ocp, default_value=default_value, t_span=t_span, x=initial_states, p=p, k=k)

    if verbose:
        print(f'Matching states, parameters, and time to boundary conditions:')
    guess = match_states_to_boundary_conditions(ocp, guess, rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)

    if verbose:
        print(f'Propagating the dynamics in time\n')
    guess = propagate_ocp_guess_from_guess(
        ocp, t_span, guess, control=control, abs_tol=abs_tol, rel_tol=rel_tol, max_step=max_step, reverse=reverse)

    if match_constants:
        if verbose:
            print(f'Matching states, parameters, and time to dynamics:')
        guess = match_constants_to_boundary_conditions(ocp, guess, rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)

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
        max_step: Optional[float] = None,
        reverse: bool = False,
        match_constants: bool = True,
        fit_adjoints: bool = True,
        quadrature: str = 'simpson',
        condition_adjoints: bool = False,
        verbose: bool = False
) -> Solution:

    guess = initialize_guess(dual, default_value=default_value, t_span=t_span, x=initial_states, lam=initial_costates,
                             p=p, nu0=nu0, nuf=nuf, k=k)

    if verbose:
        print(f'Matching states, parameters, and time to boundary conditions:')

    guess = match_states_to_boundary_conditions(dual, guess, rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)

    if verbose:
        print(f'Propagating the dynamics in time\n')

    if fit_adjoints:
        guess = propagate_ocp_guess_from_guess(
            dual, t_span, guess, control=control, abs_tol=abs_tol, rel_tol=rel_tol, max_step=max_step,
            reverse=reverse)

    else:
        guess = propagate_dual_guess_from_guess(
            dual, t_span, guess, control=control, abs_tol=abs_tol, rel_tol=rel_tol, max_step=max_step,
            reverse=reverse)

    if match_constants:
        if verbose:
            print(f'Matching the constants to the boundary conditions:')

        guess = match_constants_to_boundary_conditions(dual, guess, rel_tol=rel_tol, abs_tol=abs_tol, verbose=verbose)

    if fit_adjoints:
        if verbose:
            print(f'Fitting the costates and adjoint parameters:')

        guess.lam = process_dynamic_value(guess.lam[:, 0], guess.x.shape)
        guess = match_adjoints(dual, guess, quadrature=quadrature, rel_tol=rel_tol, abs_tol=abs_tol,
                               condition_adjoints=condition_adjoints, verbose=verbose)

    return guess
