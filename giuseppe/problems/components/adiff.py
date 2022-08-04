from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import casadi as ca


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
