from sympy import Symbol, sympify
from typing import Union, get_args, get_origin

from giuseppe.utils.typing import SymExpr


class Symbolic:
    def __init__(self):
        self.sym_locals: dict[str, SymExpr] = {}

    def new_sym(self, name: str):
        if name in self.sym_locals:
            raise ValueError(f'{name} already defined')
        elif name is None:
            raise RuntimeWarning('No varibale name given')

        sym = Symbol(name)
        self.sym_locals[name] = sym
        return sym

    def sympify(self, expr: str) -> SymExpr:
        return sympify(expr, locals=self.sym_locals)


class Picky:
    SUPPORTED_INPUTS: type = object

    def __init__(self, data_source: SUPPORTED_INPUTS):
        if get_origin(self.SUPPORTED_INPUTS) is Union:
            if not isinstance(data_source, get_args(self.SUPPORTED_INPUTS)):
                raise TypeError(f'{self.__class__} cannot ingest type {type(data_source)}')
        else:
            if not isinstance(data_source, self.SUPPORTED_INPUTS):
                raise TypeError(f'{self.__class__} cannot ingest type {type(data_source)}')
