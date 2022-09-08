from typing import Union, Tuple, TYPE_CHECKING, TypeVar

import casadi as ca

from giuseppe.problems.regularization.generic import Regularizer
from giuseppe.utils.typing import Symbol, SymExpr

if TYPE_CHECKING:
    from giuseppe.problems import AdiffOCP
    from giuseppe.problems.components.adiffInput import InputAdiffInequalityConstraint
else:
    AdiffOCP = TypeVar('AdiffOCP')
    InputAdiffInequalityConstraint = TypeVar('InputAdiffInequalityConstraint')


class AdiffControlConstraintHandler(Regularizer):
    def __init__(self, regulator: ca.MX, method: str = 'atan'):
        self.regulator: ca.MX = regulator
        self.method: str = method

        if method.lower() in ['atan', 'arctan']:
            self.expr_generator = self._gen_atan_expr
        elif method.lower() in ['trig', 'sin']:
            self.expr_generator = self._gen_trig_expr
        elif method.lower() in ['erf', 'error']:
            self.expr_generator = self._gen_erf_expr
        elif method.lower() in ['tanh']:
            self.expr_generator = self._gen_tanh_expr
        elif method.lower() in ['logistic']:
            self.expr_generator = self._gen_logistic_expr
        elif method.lower() in ['alg', 'algebraic']:
            self.expr_generator = self._gen_alg_expr
        else:
            raise ValueError(f'method ''{method}'' not implemented')

    # TODO: Add technique to compute real control automatically
    # TODO: Explore one-sided control functions
    def apply(self, prob: AdiffOCP, control_constraint: InputAdiffInequalityConstraint, position: str) -> AdiffOCP:
        if position not in ['control', 'path']:
            raise ValueError(f'Location of control constraint regularizer should be \'control\' or \'path\'')

        bounded_control = control_constraint.expr
        pseudo_control = ca.MX.sym(bounded_control.str() + '_reg', 1)

        # CasADi can't check if sym is in vector, so check if substitution changed vector instead
        pseudo_controls = ca.substitute(prob.controls, bounded_control, pseudo_control)
        if ca.is_equal(prob.controls, pseudo_controls):
            raise ValueError(f'Control {bounded_control} not found to add constraint')
        else:
            prob.controls = pseudo_controls

        if control_constraint.lower_limit is None or control_constraint.upper_limit is None:
            raise ValueError('Control constraints must have lower and upper limits')

        control_expr, error_expr = self.expr_generator(pseudo_control,
                                                       control_constraint.lower_limit, control_constraint.upper_limit,
                                                       self.regulator)

        prob.eom = ca.substitute(prob.eom, bounded_control, control_expr)

        prob.inputConstraints.initial = ca.substitute(prob.inputConstraints.initial, bounded_control, control_expr)
        prob.inputConstraints.terminal = ca.substitute(prob.inputConstraints.terminal, bounded_control, control_expr)

        prob.inputCost.initial = ca.substitute(prob.inputCost.initial, bounded_control, control_expr)
        prob.inputCost.terminal = ca.substitute(prob.inputCost.terminal, bounded_control, control_expr)

        prob.inputCost.path = ca.substitute(prob.inputCost.path, bounded_control, control_expr) + error_expr

        return prob

    @staticmethod
    def _gen_atan_expr(pseudo_control: ca.MX, lower_limit: Union[ca.MX, float], upper_limit: Union[ca.MX, float],
                       regulator: ca.MX) -> Tuple[ca.MX, ca.MX]:

        control_expr = (upper_limit - lower_limit) / ca.pi * ca.atan(pseudo_control / regulator) \
                       + (upper_limit + lower_limit) / 2
        error_expr = regulator * ca.log(1 + pseudo_control**2 / regulator**2) / ca.pi

        return control_expr, error_expr

    @staticmethod
    def _gen_trig_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) / 2 * ca.sin(pseudo_control) + (upper_limit + lower_limit) / 2
        error_expr = -regulator * ca.cos(pseudo_control)

        return control_expr, error_expr

    @staticmethod
    def _gen_erf_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) / 2 * ca.erf(pseudo_control) + (upper_limit + lower_limit) / 2
        error_expr = regulator * (1 - ca.exp(-pseudo_control**2/regulator**2)) / ca.sqrt(ca.pi)

        return control_expr, error_expr

    @staticmethod
    def _gen_tanh_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) / 2 * ca.tanh(pseudo_control / regulator) \
            + (upper_limit + lower_limit) / 2
        error_expr = pseudo_control * ca.tanh(pseudo_control / regulator) \
            - regulator * ca.log(ca.cosh(pseudo_control / regulator))

        return control_expr, error_expr

    @staticmethod
    def _gen_logistic_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) * ((1 + ca.exp(-pseudo_control / regulator))**-1 - 1 / 2) \
            + (upper_limit + lower_limit) / 2
        error_expr = - 2 * regulator * ca.log((1 + ca.exp(-pseudo_control / regulator)) / 2) \
            + 2 * pseudo_control * ((1 + ca.exp(-pseudo_control / regulator)) ** -1 - 1)

        return control_expr, error_expr

    @staticmethod
    def _gen_alg_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) / 2 * (pseudo_control / regulator) \
            * (1 + (pseudo_control ** 2 / regulator ** 2)) + (upper_limit + lower_limit) / 2
        error_expr = regulator * (1 - (1 + (pseudo_control ** 2 / regulator ** 2)) ** (-1/2))

        return control_expr, error_expr
