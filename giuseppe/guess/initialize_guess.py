from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import Problem


def initialize_guess(
        prob: Problem, default_value: float = 1., t_span: Union[float, ArrayLike] = 1.) -> Solution:
    """
    Generate guess where all variables (excluding the independent) are set to a default single constant

    Main purpose is to initialize a solution object for more advanced guess generators

    Parameters
    ----------
    prob : Problem
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

    data = {'converged': False}

    if isinstance(t_span, float) or isinstance(t_span, int):
        data['t'] = np.asarray([0., t_span], dtype=float)
    else:
        data['t'] = np.asarray(t_span, dtype=float)

    num_t_steps = len(data['t'])

    if hasattr(prob, 'num_states'):
        data['x'] = np.ones((prob.num_states, num_t_steps)) * default_value

    if hasattr(prob, 'num_parameters'):
        data['p'] = np.ones((prob.num_parameters,)) * default_value

    if hasattr(prob, 'default_values'):
        data['k'] = prob.default_values

    if hasattr(prob, 'num_controls'):
        data['u'] = np.ones((prob.num_controls, num_t_steps)) * default_value

    if hasattr(prob, 'num_costates'):
        data['lam'] = np.ones((prob.num_costates, num_t_steps)) * default_value

    if hasattr(prob, 'num_initial_adjoints'):
        data['nu0'] = np.ones((prob.num_initial_adjoints,)) * default_value
        data['nuf'] = np.ones((prob.num_terminal_adjoints,)) * default_value

    return Solution(**data)
