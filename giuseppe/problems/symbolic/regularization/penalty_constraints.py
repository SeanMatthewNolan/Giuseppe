from typing import Union, Tuple

import sympy

from giuseppe.problems.components.input import InputInequalityConstraint
from giuseppe.utils.typing import Symbol, SymExpr

from .generic import SymRegularizer, Problem


class PenaltyConstraintHandler(SymRegularizer):
    def __init__(self, regulator: Union[str, Symbol], method: str = 'sec'):
        self.regulator: Union[str, Symbol] = regulator
        self.method: str = method

        if method.lower() in ['utm', 'secant', 'sec']:
            self.expr_generator = self._gen_sec_expr
        elif method.lower() in ['rational', 'rat']:
            self.expr_generator = self._gen_rat_expr
        else:
            raise ValueError(f'method \'{method}\' not implemented')

    def apply(self, prob: Problem, constraint: InputInequalityConstraint, position: str) -> Problem:

        expr = prob.sympify(constraint.expr)
        lower_limit = prob.sympify(constraint.lower_limit)
        upper_limit = prob.sympify(constraint.upper_limit)
        regulator = prob.sympify(self.regulator)

        penalty_func = self.expr_generator(expr, lower_limit, upper_limit, regulator)

        if position.lower() == 'initial':
            prob.cost.initial += penalty_func
        elif position.lower() in ['path', 'control']:
            prob.cost.path += penalty_func
        elif position.lower() == 'terminal':
            prob.cost.terminal += penalty_func

        return prob

    @staticmethod
    def _gen_sec_expr(expr: SymExpr, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        if lower_limit is None or upper_limit is None:
            raise ValueError(f'Path constraints using UTM/secant method must have lower and upper limits')

        penalty_func = regulator \
                       / sympy.cos(
            sympy.pi / 2 * (2 * expr - upper_limit - lower_limit) / (upper_limit - lower_limit)) - regulator
        return penalty_func

    @staticmethod
    def _gen_rat_expr(expr: SymExpr, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        if lower_limit is not None and upper_limit is not None:
            penalty_func = regulator \
                           * (1 / (expr - lower_limit) + 1 / (upper_limit - expr) + 4 / (lower_limit - upper_limit))
        elif lower_limit is not None:
            penalty_func = regulator / (expr - lower_limit)
        elif upper_limit is not None:
            penalty_func = regulator / (upper_limit - expr)
        else:
            raise ValueError(f'Lower or upper limit must be specificed for inequality path constraint.')

        return penalty_func
