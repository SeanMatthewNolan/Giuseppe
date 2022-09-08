from typing import Union, Tuple, TYPE_CHECKING, TypeVar

import sympy
import casadi as ca

from giuseppe.problems.regularization.generic import Regularizer
from giuseppe.utils.typing import Symbol, SymExpr

if TYPE_CHECKING:
    from giuseppe.problems import AdiffOCP
    from giuseppe.problems.components.adiffInput import InputAdiffInequalityConstraint
else:
    AdiffOCP = TypeVar('AdiffOCP')
    InputAdiffInequalityConstraint = TypeVar('InputAdiffInequalityConstraint')


class AdiffPenaltyConstraintHandler(Regularizer):
    def __init__(self, regulator: ca.MX, method: str = 'sec'):
        self.regulator: ca.MX = regulator
        self.method: str = method

        if method.lower() in ['utm', 'secant', 'sec']:
            self.expr_generator = self._gen_sec_expr
        elif method.lower() in ['rational', 'rat']:
            self.expr_generator = self._gen_rat_expr
        else:
            raise ValueError(f'method \'{method}\' not implemented')

    def apply(self, prob: AdiffOCP, constraint: InputAdiffInequalityConstraint, position: str) -> AdiffOCP:

        penalty_func = self.expr_generator(constraint.expr, constraint.lower_limit, constraint.upper_limit,
                                           self.regulator)

        if position.lower() == 'initial':
            prob.inputCost.initial += penalty_func
        elif position.lower() in ['path', 'control']:
            prob.inputCost.path += penalty_func
        elif position.lower() == 'terminal':
            prob.inputCost.terminal += penalty_func

        return prob

    @staticmethod
    def _gen_sec_expr(expr: ca.MX, lower_limit: Union[ca.MX, float], upper_limit: Union[ca.MX, float],
                      regulator: ca.MX) -> ca.MX:

        if lower_limit is None or upper_limit is None:
            raise ValueError(f'Path constraints using UTM/secant method must have lower and upper limits')

        penalty_func = regulator / ca.cos(ca.pi / 2 * (2 * expr - upper_limit - lower_limit)
                                          / (upper_limit - lower_limit)) - regulator
        return penalty_func

    @staticmethod
    def _gen_rat_expr(expr: ca.MX, lower_limit: Union[ca.MX, float], upper_limit: Union[ca.MX, float],
                      regulator: ca.MX) -> ca.MX:

        if lower_limit is not None and upper_limit is not None:
            penalty_func = regulator * (1 / (expr - lower_limit) + 1 / (upper_limit - expr)
                                        + 4 / (lower_limit - upper_limit))
        elif lower_limit is not None:
            penalty_func = regulator / (expr - lower_limit)
        elif upper_limit is not None:
            penalty_func = regulator / (upper_limit - expr)
        else:
            raise ValueError(f'Lower or upper limit must be specified for inequality path constraint.')

        return ca.MX.sym(penalty_func)
