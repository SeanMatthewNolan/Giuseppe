from warnings import warn
from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.problems.dual.solution import Solution
from giuseppe.problems.dual.utils import sift_ocp_and_dual
from giuseppe.problems.ocp import CompOCP
from giuseppe.problems.typing import AnyProblem, AnySolution


def generate_single_constant_guess(comp_prob: AnyProblem, constant: float = 1., t_span: Union[float, ArrayLike] = 0.1) \
        -> AnySolution:
    """
    Generate guess where all variables (excluding the indenpendent) are set to a single constant

    Main purpose is to initialize a solution object for more advanced guess generators

    Parameters
    ----------
    comp_prob : CompBVP, CompOCP, CompDual or CompDualOCP
        the problem that the guess is for, needed to shape/size of arrays

    constant : float, default=1.
        the constant all variables, except time (t_span) and constants (problem's default_values), are set to

    t_span : float or ArrayLike, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])

    Returns
    -------
    guess : BVPSol, OCPSol, DualSol, or DualOCPSol

    """

    prob, dual = sift_ocp_and_dual(comp_prob)

    data = {'converged': False}

    if isinstance(t_span, float) or isinstance(t_span, int):
        data['t'] = np.array([0., t_span], dtype=float)
    else:
        data['t'] = np.array(t_span, dtype=float)

    num_t_steps = len(data['t'])

    if prob is not None:
        data['x'] = np.ones((prob.num_states, num_t_steps)) * constant
        data['p'] = np.ones((prob.num_parameters,)) * constant
        data['k'] = prob.default_values

    if isinstance(prob, CompOCP):
        data['u'] = np.ones((prob.num_controls, num_t_steps)) * constant

    if dual is not None:
        data['lam'] = np.ones((dual.num_costates, num_t_steps)) * constant
        data['nu0'] = np.ones((dual.num_initial_adjoints,)) * constant
        data['nuf'] = np.ones((dual.num_terminal_adjoints,)) * constant

    return Solution(**data)


def update_value_constant(guess: AnySolution, name: str, values: Optional[Union[float, ArrayLike]]) -> AnySolution:
    """
    Update values in guess to given constant values

    Parameters
    ----------
    guess : BVPSol, OCPSol, DualSol, or DualOCPSol
        guess to update with values
    name : str
        name of attribute in guess to update
    values : ArrayLike or float, optional, default=None
        value or values to set attribute to

    Returns
    -------
    BVPSol, OCPSol, DualSol, or DualOCPSol

    """
    if values is None or not hasattr(guess, name):
        return guess
    else:
        current_values = np.asarray(getattr(guess, name))

    if isinstance(values, float) or isinstance(values, int):
        values = values * np.ones_like(current_values)
    else:
        values = np.asarray(values, dtype=float)
        if current_values.ndim == 2:
            values = np.vstack([values for _ in guess.t]).T

    if values.shape != current_values.shape:
        warn(f'Changing size of "{name}" from {current_values.shape} to {values.shape}')

    setattr(guess, name, values)
    return guess


def generate_constant_guess(
        comp_prob: AnyProblem, default_value: float = 0.1, t_span: Union[float, ArrayLike] = 0.1,
        x: Optional[Union[float, ArrayLike]] = None, lam: Optional[Union[float, ArrayLike]] = None,
        u: Optional[Union[float, ArrayLike]] = None, p: Optional[Union[float, ArrayLike]] = None,
        nu_0: Optional[Union[float, ArrayLike]] = None, nu_f: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None) -> AnySolution:
    """
    Generate guess where variables (excluding the indenpendent) are set to constant values

    Main purpose is to initialize a solution object for more advanced guess generators

    Parameters
    ----------
    comp_prob : CompBVP, CompOCP, CompDual or CompDualOCP
        the problem that the guess is for, needed to shape/size of arrays
    default_value : float, default=0.1
        value used if no value is given
    t_span : ArrayLike or float, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])
    x : ArrayLike or float, optional, default=None
        state values
    lam  : ArrayLike or float, optional, default=None
        costate values
    u : ArrayLike or float, optional, default=None
        control values
    p : ArrayLike or float, optional, default=None
        parameter values
    nu_0 : ArrayLike or float, optional, default=None
        initial adjoint values
    nu_f : ArrayLike or float, optional, default=None
        terminal adjoint values
    k : ArrayLike or float, optional, default=None
        constants values

    Returns
    -------
    guess : BVPSol, OCPSol, DualSol, or DualOCPSol

    """

    guess = generate_single_constant_guess(comp_prob, constant=default_value, t_span=t_span)

    update_value_constant(guess, 'x', x)
    update_value_constant(guess, 'lam', lam)
    update_value_constant(guess, 'u', u)
    update_value_constant(guess, 'p', p)
    update_value_constant(guess, 'nu_0', nu_0)
    update_value_constant(guess, 'nu_f', nu_f)
    update_value_constant(guess, 'k', k)

    return guess
