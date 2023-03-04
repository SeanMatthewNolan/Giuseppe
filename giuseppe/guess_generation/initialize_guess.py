import copy
from typing import Union, Optional
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import Problem


def process_static_value(input_value: Union[float, ArrayLike], output_len: int):

    input_dim = np.ndim(input_value)
    if input_dim == 0:
        output_array = np.empty((output_len,), dtype=float)
        output_array.fill(input_value)
    elif input_dim == 1:
        output_array = np.asarray(input_value, dtype=float)
    else:
        raise ValueError('Given input_value has more than 1 dimensions')

    if len(output_array) != output_len:
        raise ValueError(f'Cannot match input with shape {len(input_value)} to specified shape {output_len}')

    return output_array


def process_dynamic_value(input_value: Union[float, ArrayLike], output_shape: tuple[int, int]):

    input_dim = np.ndim(input_value)
    if input_dim == 0:
        output_array = np.empty(output_shape, dtype=float)
        output_array.fill(input_value)
    elif input_dim == 1:
        output_array = np.tile(np.reshape(input_value, (output_shape[0], -1)), output_shape[1])
    elif input_dim == 2:
        output_array = np.asarray(input_value, dtype=float)
    else:
        raise ValueError('Given input_value has more than 2 dimensions')

    if output_array.shape != output_shape:
        raise ValueError(f'Cannot match input with shape {np.shape(input_value)} to specified shape {output_shape}')

    return output_array


def initialize_guess(
        prob: Problem,
        default_value: float = 1.,
        t_span: Union[float, ArrayLike] = 1.,
        x: Optional[Union[ArrayLike, float]] = None,
        p: Optional[Union[ArrayLike, float]] = None,
        u: Optional[Union[ArrayLike, float]] = None,
        k: Optional[Union[float, ArrayLike]] = None,
        lam: Optional[Union[ArrayLike, float]] = None,
        nu0: Optional[Union[ArrayLike, float]] = None,
        nuf: Optional[Union[ArrayLike, float]] = None,
) -> Solution:

    """
    Generate guess where all variables (excluding the independent) are set to a default single constant

    Main purpose is to initialize a solution object for more advanced guess generators

    Parameters
    ----------
    prob : Problem
        the problem that the guess is for, needed to shape/size of arrays

    default_value : float, default=1.
        the constant all variables, except time (t_span) and constants (problem's default_values), are set to if not
         otherwise specified

    t_span : float or ArrayLike, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])

    x : NDArray, Optional
        state values to initialize guess with

    p : NDArray, Optional
        parameter values to initialize guess with

    u : NDArray, Optional
        control values to initialize guess with

    k : NDArray, Optional
        constant values to initialize guess with

    lam : NDArray, Optional
        costate values to initialize guess with

    nu0 : NDArray, Optional
        initial adjoint parameter values to initialize guess with

    nuf : NDArray, Optional
        terminal adjoint parameter values to initialize guess with

    Returns
    -------
    guess : Solution

    """

    data = {'converged': False, 'annotations': deepcopy(prob.annotations)}

    if isinstance(t_span, (float, int)):
        data['t'] = np.asarray([0., t_span], dtype=float)
    else:
        data['t'] = np.asarray(t_span, dtype=float)

    num_t_steps = len(data['t'])

    if hasattr(prob, 'num_states'):
        if x is None:
            x = default_value

        data['x'] = process_dynamic_value(x, (prob.num_states, num_t_steps))

    if hasattr(prob, 'num_parameters'):
        if p is None:
            p = default_value

        data['p'] = process_static_value(p, prob.num_parameters)

    if hasattr(prob, 'num_constants'):
        if k is None:
            k = prob.default_values

        data['k'] = process_static_value(k, prob.num_constants)

    if hasattr(prob, 'num_controls'):
        if u is None:
            u = default_value

        data['u'] = process_dynamic_value(u, (prob.num_controls, num_t_steps))

    if hasattr(prob, 'num_costates'):
        if lam is None:
            lam = default_value

        data['lam'] = process_dynamic_value(lam, (prob.num_costates, num_t_steps))

    if hasattr(prob, 'num_initial_adjoints'):
        if nu0 is None:
            nu0 = default_value
        if nuf is None:
            nuf = default_value

        data['nu0'] = process_static_value(nu0, prob.num_initial_adjoints)
        data['nuf'] = process_static_value(nuf, prob.num_terminal_adjoints)

    return Solution(**data)


def initialize_guess_from_partial_solution(
        prob: Problem,
        partial_solution: Solution,
        default_value: float = 1.,
) -> Solution:

    guess = initialize_guess(prob, default_value=default_value)

    if partial_solution.t is not None:
        guess.t = copy.copy(partial_solution.t)

    if partial_solution.x is not None:
        guess.x = copy.copy(partial_solution.x)

    if partial_solution.p is not None:
        guess.p = copy.copy(partial_solution.p)

    if partial_solution.k is not None:
        guess.k = copy.copy(partial_solution.k)

    if partial_solution.u is not None:
        guess.u = copy.copy(partial_solution.u)

    if partial_solution.p is not None:
        guess.p = copy.copy(partial_solution.p)

    if partial_solution.lam is not None:
        guess.lam = copy.copy(partial_solution.lam)

    if partial_solution.nu0 is not None:
        guess.nu0 = copy.copy(partial_solution.nu0)

    if partial_solution.nuf is not None:
        guess.nuf = copy.copy(partial_solution.nuf)

    if partial_solution.aux is not None:
        guess.aux = copy.copy(partial_solution.aux)

    return guess




