from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

import casadi as ca
import numpy as np

from giuseppe.utils.conversion import ca_vec2arr, ca_mat2arr


@dataclass
class AdiffBoundaryConditions:
    initial: ca.Function
    terminal: ca.Function


@dataclass
class AdiffCost:
    initial: ca.Function
    path: ca.Function
    terminal: ca.Function


def ca_wrap(name: str, ca_args: Iterable, function: Callable, func_args: Iterable,
            ca_arg_names: Iterable, out_name: Optional[str] = None) -> ca.Function:
    if out_name is None:
        out_name = name

    if hasattr(function(*func_args), '__len__'):
        expression = function(*func_args)
    else:
        expression = (function(*func_args),)

    return ca.Function(name,
                       ca_args,
                       (ca.vcat(expression),),
                       ca_arg_names,
                       (out_name,))


def lambdify_ca(fun: ca.Function):
    if fun.size_out(0)[1] == 1:
        return lambda *args: ca_vec2arr(fun(*args))
    else:
        return lambda *args: ca_mat2arr(fun(*args))
