from typing import Union, Tuple, TYPE_CHECKING, TypeVar

import sympy

from giuseppe.problems.regularization.generic import Regularizer
from giuseppe.utils.typing import Symbol, SymExpr

if TYPE_CHECKING:
    from giuseppe.problems import SymOCP
    from giuseppe.problems.components.input import InputInequalityConstraint
else:
    SymOCP = TypeVar('SymOCP')
    InputInequalityConstraint = TypeVar('InputInequalityConstraint')


class PenaltyConstraintHandler(Regularizer):
    def __init__(self, regulator: Union[str, Symbol], method: str = 'sec'):
        self.regulator: Union[str, Symbol] = regulator
        self.method: str = method

        if method.lower() in ['utm', 'sec']:
            self.expr_generator = self._gen_sec_expr
        else:
            raise ValueError(f'method \'{method}\' not implemented')

    def apply(self, prob: SymOCP, constraint: InputInequalityConstraint, position: str) -> SymOCP:

        expr = prob.sympify(constraint.expr)
        lower_limit = prob.sympify(constraint.lower_limit)
        upper_limit = prob.sympify(constraint.upper_limit)
        regulator = prob.sympify(self.regulator)

        if lower_limit is None or upper_limit is None:
            raise ValueError(f'Path constraints using \'{self.method}\' must have lower and upper limits')

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
        penalty_func = regulator \
                       / sympy.cos(sympy.pi / 2 * (2 * expr - upper_limit - lower_limit) / (upper_limit - lower_limit))
        return penalty_func
