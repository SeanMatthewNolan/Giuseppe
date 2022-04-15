from warnings import warn
from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.problems.typing import AnyProblem, AnySolution
from ..constant import update_constant_value, initialize_guess_w_default_value


def update_linear_value(guess: AnySolution, name: str, initial_values: Optional[Union[float, ArrayLike]],
                        terminal_values: Optional[Union[float, ArrayLike]]) -> AnySolution:
    """
    Update dyanic values in guess to linearly change from initial values to terminal values over the trajectory

    Parameters
    ----------
    guess : BVPSol, OCPSol, DualSol, or DualOCPSol
        guess to update with values
    name : str
        name of attribute in guess to update
    initial_values : ArrayLike or float, optional, default=None
        value or values to set at beginning of the trajectory
    terminal_values : ArrayLike or float, optional, default=None
        value or values to set at beginning of the trajectory

    Returns
    -------
    BVPSol, OCPSol, DualSol, or DualOCPSol

    """
    if not hasattr(guess, name):
        return guess
    else:
        current_values = np.asarray(getattr(guess, name))

    if current_values.ndim == 1:
        warn(f'Attempting to update {name} which is constant through trajectory! Setting to specified initial value')
        update_constant_value(guess, name, initial_values)

    if initial_values is None:
        update_constant_value(guess, name, terminal_values)
    elif terminal_values is None:
        update_constant_value(guess, name, initial_values)
    else:
        if isinstance(initial_values, float) or isinstance(initial_values, int):
            initial_values = initial_values * np.ones_like(current_values[:, 0])
        else:
            initial_values = np.asarray(initial_values, dtype=float)

        if isinstance(terminal_values, float) or isinstance(terminal_values, int):
            terminal_values = terminal_values * np.ones_like(current_values[:, -1])
        else:
            terminal_values = np.asarray(terminal_values, dtype=float)

        values = np.linspace(initial_values, terminal_values, len(guess.t)).T

        if values.shape != current_values.shape:
            warn(f'Changing size of "{name}" from {current_values.shape} to {values.shape}')

        setattr(guess, name, values)
        return guess


def generate_linear_guess(
        comp_prob: AnyProblem, default_value: float = 0.1, t_span: Union[float, ArrayLike] = 0.1,
        x0: Optional[Union[float, ArrayLike]] = None, xf: Optional[Union[float, ArrayLike]] = None,
        lam0: Optional[Union[float, ArrayLike]] = None, lamf: Optional[Union[float, ArrayLike]] = None,
        u0: Optional[Union[float, ArrayLike]] = None, uf: Optional[Union[float, ArrayLike]] = None,
        p: Optional[Union[float, ArrayLike]] = None, nu_0: Optional[Union[float, ArrayLike]] = None,
        nu_f: Optional[Union[float, ArrayLike]] = None, k: Optional[Union[float, ArrayLike]] = None) -> AnySolution:
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
    x0 : ArrayLike or float, optional, default=None
        initial state values
    xf : ArrayLike or float, optional, default=None
        terminal state values
    lam0 : ArrayLike or float, optional, default=None
        initial costate values
    lamf : ArrayLike or float, optional, default=None
        terminal costate values
    u0 : ArrayLike or float, optional, default=None
        initial control values
    uf : ArrayLike or float, optional, default=None
        terminal control values
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

    guess = initialize_guess_w_default_value(comp_prob, default_value=default_value, t_span=t_span)

    update_linear_value(guess, 'x', x0, xf)
    update_linear_value(guess, 'lam', lam0, lamf)
    update_linear_value(guess, 'u', u0, uf)
    update_constant_value(guess, 'p', p)
    update_constant_value(guess, 'nu_0', nu_0)
    update_constant_value(guess, 'nu_f', nu_f)
    update_constant_value(guess, 'k', k)

    return guess
