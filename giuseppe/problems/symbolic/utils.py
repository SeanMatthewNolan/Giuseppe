from sympy import Symbol, sympify

from giuseppe.utils.typing import SymExpr


class Symbolic:
    def __init__(self):
        self.sym_locals: dict[str, SymExpr] = {}

    def new_sym(self, name: str):
        if name in self.sym_locals:
            raise ValueError(f'{name} already defined')
        elif name is None:
            raise RuntimeWarning('No variable name given')

        sym = Symbol(name)
        self.sym_locals[name] = sym
        return sym

    def sympify(self, expr: str) -> SymExpr:
        return sympify(expr, locals=self.sym_locals)
