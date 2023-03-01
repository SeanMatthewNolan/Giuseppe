from typing import Union, Sequence, Callable, Optional
from warnings import warn

import numba
from numba.core.errors import NumbaError
from sympy import lambdify as sympy_lambdify, Expr
from sympy.utilities.iterables import flatten

JIT_COMPILE = True
CSE = True            # Compilation faster when true because Numba does not have to perform
EAGER_COMPILE = True  # As of 3/1/23 Eager compilation appears actually faster overall
PRINT_LAMBDA_FUNCS = False
LAMB_MODS = ['numpy', 'math']


# TODO Find suitable type for expr
def lambdify(args: Union[Sequence, Expr], expr, flatten_args: bool = False, use_jit_compile: bool = True):
    if flatten_args:
        args = flatten(args)

    func = sympy_lambdify(args, expr, cse=CSE, modules=LAMB_MODS)

    if PRINT_LAMBDA_FUNCS:
        from inspect import getsource
        print(getsource(func))

    if use_jit_compile:
        func = jit_compile(func, signature=signature_from_args(args))

    return func


def signature_from_args(args: Union[Sequence, Expr]):
    return tuple([numba.float64[:] if isinstance(arg, Sequence) or hasattr(arg, '__array__')
                  else numba.float64 for arg in args])


# TODO: get more specific type for "signature"
# (see https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#numba.jit in meantime)
def jit_compile(func: Callable, signature: Optional = None):
    if not JIT_COMPILE:
        return func

    try:
        if (signature is not None) and EAGER_COMPILE:
            return numba.njit(signature)(func)
        else:
            return numba.njit()(func)

    except NumbaError as e:
        warn(f'Numba error {e} prevented compilation of function {func}. Uncompiled version will be used.')
        return func
