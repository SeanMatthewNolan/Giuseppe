from sympy import Symbol, sympify

from giuseppe.utils.aliases import SymExpr, SymMatrix, EMPTY_SYM_MATRIX, SYM_NULL, SYM_ZERO


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
