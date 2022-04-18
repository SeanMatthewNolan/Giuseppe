from dataclasses import dataclass
import casadi as ca
from typing import Callable


@dataclass
class AdiffBoundaryConditions:
    initial: ca.Function
    terminal: ca.Function


@dataclass
class AdiffCost:
    initial: ca.Function
    path: ca.Function
    terminal: ca.Function


def wrap_func(name: str, ca_args, function: Callable, func_args, ca_arg_names, out_name: str = None) -> ca.Function:
    if out_name is None:
        out_name = name

    return ca.Function(name,
                       ca_args,
                       (ca.vcat(function(*func_args)),),
                       ca_arg_names,
                       (out_name,))
