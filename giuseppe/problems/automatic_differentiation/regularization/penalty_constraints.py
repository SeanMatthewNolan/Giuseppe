from typing import Union, TYPE_CHECKING, TypeVar

import casadi as ca

from giuseppe.problems.protocols.regularizer import Regularizer

if TYPE_CHECKING:
    from ..ocp import ADiffOCP
    from ..input import ADiffInputInequalityConstraint
else:
    ADiffOCP = TypeVar('ADiffOCP')
    ADiffInputInequalityConstraint = TypeVar('ADiffInputInequalityConstraint')


class ADiffPenaltyConstraintHandler(Regularizer):
    def __init__(self, regulator: Union[ca.SX, ca.MX], method: str = 'sec'):
        self.regulator: Union[ca.SX, ca.MX] = regulator
        self.method: str = method

        if method.lower() in ['utm', 'secant', 'sec']:
            self.expr_generator = self._gen_sec_expr
        elif method.lower() in ['rational', 'rat']:
            self.expr_generator = self._gen_rat_expr
        else:
            raise ValueError(f'method \'{method}\' not implemented')

    def apply(self, prob: ADiffOCP, constraint: ADiffInputInequalityConstraint, position: str) -> ADiffOCP:

        penalty_func = self.expr_generator(
                constraint.expr, constraint.lower_limit, constraint.upper_limit, self.regulator)

        if position.lower() == 'initial':
            prob.input_cost.initial += penalty_func
        elif position.lower() in ['path', 'control']:
            prob.input_cost.path += penalty_func
        elif position.lower() == 'terminal':
            prob.input_cost.terminal += penalty_func

        return prob

    @staticmethod
    def _gen_sec_expr(expr: Union[ca.SX, ca.MX], lower_limit: Union[Union[ca.SX, ca.MX], float],
                      upper_limit: Union[Union[ca.SX, ca.MX], float],
                      regulator: Union[ca.SX, ca.MX]) -> Union[ca.SX, ca.MX]:

        if lower_limit is None or upper_limit is None:
            raise ValueError(f'Path constraints using UTM/secant method must have lower and upper limits')

        penalty_func = regulator / ca.cos(ca.pi / 2 * (2 * expr - upper_limit - lower_limit)
                                          / (upper_limit - lower_limit)) - regulator
        return penalty_func

    @staticmethod
    def _gen_rat_expr(expr: Union[ca.SX, ca.MX], lower_limit: Union[ca.SX, ca.MX, float],
                      upper_limit: Union[ca.SX, ca.MX, float],
                      regulator: Union[ca.SX, ca.MX]) -> Union[ca.SX, ca.MX, float]:

        if lower_limit is not None and upper_limit is not None:
            penalty_func = regulator * (1 / (expr - lower_limit) + 1 / (upper_limit - expr)
                                        + 4 / (lower_limit - upper_limit))
        elif lower_limit is not None:
            penalty_func = regulator / (expr - lower_limit)
        elif upper_limit is not None:
            penalty_func = regulator / (upper_limit - expr)
        else:
            raise ValueError(f'Lower or upper limit must be specified for inequality path constraint.')

        return penalty_func
