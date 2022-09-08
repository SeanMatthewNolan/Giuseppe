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


class LookupTable(ca.Callback):
    ARR_INDEX = Union[list[np.ndarray], tuple[np.ndarray]]
    STR_INDEX = Union[list[str], tuple[str]]

    def __init__(self, breakpoints: ARR_INDEX, data: np.ndarray,
                 breakpoint_names: Optional[STR_INDEX] = None,
                 interp_method: str = 'bspline', name: str = 'lookup_table', opts: Optional[dict] = None):
        ca.Callback.__init__(self)
        self.breakpoints = breakpoints
        self.data = data
        self.interpolate = ca.interpolant('thrust_table', interp_method, breakpoints, data.ravel(order='F'))
        if breakpoint_names is None:
            self.breakpoint_names = [f'i{idx}' for idx in range(len(breakpoints))]
        else:
            self.breakpoint_names = breakpoint_names
        if opts is None:
            opts = {}
        self.construct(name, opts)

    def get_n_in(self):
        return len(self.breakpoint_names)

    @staticmethod
    def get_n_out():
        return 1

    @staticmethod
    def init():
        print('initializing object')

    def eval(self, *args):
        args
        return [self.interpolate(ca.vcat(*args))]


# TODO remove this example Callback
class MyCallback(ca.Callback):
    def __init__(self, name, d, opts={}):
        ca.Callback.__init__(self)
        self.d = d
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        x = arg[0]
        f = ca.sin(self.d*x)
        return [f]
