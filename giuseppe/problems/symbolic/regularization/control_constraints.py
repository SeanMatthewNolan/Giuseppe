from typing import Union, Tuple

import sympy

from giuseppe.problems.components.input import InputInequalityConstraint
from giuseppe.utils.typing import Symbol, SymExpr

from .generic import SymRegularizer, Problem


class ControlConstraintHandler(SymRegularizer):
    def __init__(self, regulator: Union[str, Symbol], method: str = 'atan'):
        self.regulator: Union[str, Symbol] = regulator
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
    def apply(self, prob: Problem, control_constraint: InputInequalityConstraint, position: str) -> Problem:
        if position not in ['control', 'path']:
            raise ValueError(f'Location of control constraint regularizer should be \'control\' or \'path\'')

        bounded_control = prob.sympify(control_constraint.expr)
        if bounded_control not in prob.controls:
            raise ValueError(f'Control {bounded_control} not found to add constraint')

        pseudo_control = prob.new_sym(f'_{control_constraint.expr}_reg')
        lower_limit = prob.sympify(control_constraint.lower_limit)
        upper_limit = prob.sympify(control_constraint.upper_limit)
        regulator = prob.sympify(self.regulator)

        if lower_limit is None or upper_limit is None:
            raise ValueError('Control constraints must have lower and upper limits')

        control_func, error_func = self.expr_generator(pseudo_control, lower_limit, upper_limit, regulator)

        prob.controls = prob.controls.subs(bounded_control, pseudo_control)

        prob.dynamics = prob.dynamics.subs(bounded_control, control_func)

        prob.boundary_conditions.initial = prob.boundary_conditions.initial.subs(bounded_control, control_func)
        prob.boundary_conditions.terminal = prob.boundary_conditions.terminal.subs(bounded_control, control_func)

        prob.cost.initial = prob.cost.initial.subs(bounded_control, control_func)
        prob.cost.terminal = prob.cost.terminal.subs(bounded_control, control_func)

        prob.cost.path = prob.cost.path.subs(bounded_control, control_func) + error_func

        for idx, expr in enumerate(prob.expressions):
            prob.expressions[idx].expr = expr.expr.subs(bounded_control, control_func)

        return prob

    @staticmethod
    def _gen_atan_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_func = (upper_limit - lower_limit) / sympy.pi * sympy.atan(pseudo_control / regulator) \
                       + (upper_limit + lower_limit) / 2
        error_func = regulator * sympy.log(1 + pseudo_control ** 2 / regulator ** 2) / sympy.pi

        return control_func, error_func

    @staticmethod
    def _gen_trig_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_func = (upper_limit - lower_limit) / 2 * sympy.sin(pseudo_control) + (upper_limit + lower_limit) / 2
        error_func = -regulator * sympy.cos(pseudo_control)

        return control_func, error_func

    @staticmethod
    def _gen_erf_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_func = (upper_limit - lower_limit) / 2 * sympy.erf(pseudo_control) + (upper_limit + lower_limit) / 2
        error_func = regulator * (1 - sympy.exp(-pseudo_control ** 2 / regulator ** 2)) / sympy.sqrt(sympy.pi)

        return control_func, error_func

    @staticmethod
    def _gen_tanh_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_func = (upper_limit - lower_limit) / 2 * sympy.tanh(pseudo_control / regulator) \
                       + (upper_limit + lower_limit) / 2
        error_func = pseudo_control * sympy.tanh(pseudo_control / regulator) \
                     - regulator * sympy.log(sympy.cosh(pseudo_control / regulator))

        return control_func, error_func

    @staticmethod
    def _gen_logistic_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_func = (upper_limit - lower_limit) * ((1 + sympy.exp(-pseudo_control / regulator)) ** -1 - 1 / 2) \
                       + (upper_limit + lower_limit) / 2
        error_func = - 2 * regulator * sympy.log((1 + sympy.exp(-pseudo_control / regulator)) / 2) \
                     + 2 * pseudo_control * ((1 + sympy.exp(-pseudo_control / regulator)) ** -1 - 1)

        return control_func, error_func

    @staticmethod
    def _gen_alg_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_func = (upper_limit - lower_limit) / 2 * (pseudo_control / regulator) \
                       * (1 + (pseudo_control ** 2 / regulator ** 2)) + (upper_limit + lower_limit) / 2
        error_func = regulator * (1 - (1 + (pseudo_control ** 2 / regulator ** 2)) ** (-1 / 2))

        return control_func, error_func
