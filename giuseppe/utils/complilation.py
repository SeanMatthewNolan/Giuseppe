from typing import Union, Sequence, Callable, Optional
from warnings import warn

import numba
from numba.core.errors import NumbaRuntimeError
from sympy import lambdify as sympy_lambdify, Expr
from sympy.utilities.iterables import flatten

JIT_COMPILE = True
CSE = True
EAGER_COMPILE = True
NUMBA_CACHE = False
LAMB_MODS = ['numpy']


# TODO Find suitable type for expr
def lambdify(args: Union[Sequence, Expr], expr, flatten_args: bool = False, compile_: bool = True):
    if flatten_args:
        args = flatten(args)

    func = sympy_lambdify(args, expr, cse=CSE, modules=LAMB_MODS)

    if compile_:
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
            return numba.njit(signature, cache=NUMBA_CACHE)(func)
        else:
            return numba.njit(cache=NUMBA_CACHE)(func)

    except NumbaRuntimeError as e:
        warn(f'Numba error {e} prevented complilation of function {func}. Uncompiled version will be used.')
        return func


# def check_for_array_arg(arg: Any):
#     return isinstance(arg, Sequence) or hasattr(arg, '__array__')
#
#
# def form_signature_from_args(args: Union[Sequence, Expr], expr: Optional[Expr] = None):
#     signature = [numba.float64[:] if check_for_array_arg(arg) else numba.float64 for arg in args]
#
#     if expr is None:
#         return signature
#     elif check_for_array_arg(expr):
#         return numba.float64[:](signature)
#     else:
#         return numba.float64(signature)
