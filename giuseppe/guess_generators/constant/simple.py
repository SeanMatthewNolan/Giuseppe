from typing import Union, Optional
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.problems.dual.utils import sift_ocp_and_dual
from giuseppe.problems.ocp import CompOCP, AdiffOCP
from giuseppe.problems.typing import AnyProblem
from ...data import Solution


def initialize_guess_w_default_value(
        comp_prob: AnyProblem, default_value: float = 1., t_span: Union[float, ArrayLike] = 0.1) -> Solution:
    """
    Generate guess where all variables (excluding the indenpendent) are set to a default single constant

    Main purpose is to initialize a solution object for more advanced guess generators

    Parameters
    ----------
    comp_prob : CompBVP, CompOCP, CompDual or CompDualOCP
        the problem that the guess is for, needed to shape/size of arrays

    default_value : float, default=1.
        the constant all variables, except time (t_span) and constants (problem's default_values), are set to

    t_span : float or ArrayLike, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])

    Returns
    -------
    guess : Solution

    """

    prob, dual = sift_ocp_and_dual(comp_prob)

    data = {'converged': False}

    if isinstance(t_span, float) or isinstance(t_span, int):
        data['t'] = np.array([0., t_span], dtype=float)
    else:
        data['t'] = np.array(t_span, dtype=float)

    num_t_steps = len(data['t'])

    if prob is not None:
        data['x'] = np.ones((prob.num_states, num_t_steps)) * default_value
        data['p'] = np.ones((prob.num_parameters,)) * default_value
        data['k'] = prob.default_values

    if isinstance(prob, CompOCP) or isinstance(prob, AdiffOCP):
        data['u'] = np.ones((prob.num_controls, num_t_steps)) * default_value

    if dual is not None:
        data['lam'] = np.ones((dual.num_costates, num_t_steps)) * default_value
        data['nu0'] = np.ones((dual.num_initial_adjoints,)) * default_value
        data['nuf'] = np.ones((dual.num_terminal_adjoints,)) * default_value

    return Solution(**data)


def update_constant_value(guess: Solution, name: str, values: Optional[Union[float, ArrayLike]]) -> Solution:
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
    Solution

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
        prob: AnyProblem, default_value: float = 0.1, t_span: Union[float, ArrayLike] = 0.1,
        x: Optional[Union[float, ArrayLike]] = None, lam: Optional[Union[float, ArrayLike]] = None,
        u: Optional[Union[float, ArrayLike]] = None, p: Optional[Union[float, ArrayLike]] = None,
        nu_0: Optional[Union[float, ArrayLike]] = None, nu_f: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None) -> Solution:
    """
    Generate guess where variables (excluding the indenpendent) are set to constant values

    Main purpose is to initialize a solution object for more advanced guess generators

    Parameters
    ----------
    prob : CompBVP, CompOCP, CompDual or CompDualOCP
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
    guess : Solution

    """

    guess = initialize_guess_w_default_value(prob, default_value=default_value, t_span=t_span)

    update_constant_value(guess, 'x', x)
    update_constant_value(guess, 'lam', lam)
    update_constant_value(guess, 'u', u)
    update_constant_value(guess, 'p', p)
    update_constant_value(guess, 'nu_0', nu_0)
    update_constant_value(guess, 'nu_f', nu_f)
    update_constant_value(guess, 'k', k)

    return guess
