from typing import Union, Optional, Tuple

import numpy as np
import sympy

from giuseppe.data_classes.solution import Solution
from giuseppe.utils.typing import Symbol, SymExpr
from giuseppe.utils.compilation import lambdify

from ..input import StrInputInequalityConstraint
from .generic import SymRegularizer, Problem


class PenaltyConstraintHandler(SymRegularizer):
    def __init__(self, regulator: Union[str, Symbol], method: str = 'sec'):
        self.regulator: Union[str, Symbol] = regulator
        self.method: str = method

        if method.lower() in ['utm', 'secant', 'sec']:
            self.expr_generator = self._gen_sec_expr
        elif method.lower() in ['rational', 'rat']:
            self.expr_generator = self._gen_rat_expr
        elif method.lower() in ['exterior', 'cubic', 'cube']:
            self.expr_generator = self._gen_cubic_expr
        else:
            raise ValueError(f'method \'{method}\' not implemented')

        self._penalty_func: Optional[SymExpr] = None
        self._constraint: Optional[StrInputInequalityConstraint] = None

    def apply(self, prob: Problem, constraint: StrInputInequalityConstraint, position: str) -> Problem:

        expr = prob.sympify(constraint.expr)
        lower_limit = prob.sympify(constraint.lower_limit)
        upper_limit = prob.sympify(constraint.upper_limit)
        regulator = prob.sympify(self.regulator)

        self._penalty_func = self.expr_generator(expr, lower_limit, upper_limit, regulator)
        self._constraint = constraint

        if position.lower() == 'initial':
            prob.cost.initial += self._penalty_func
        elif position.lower() in ['path', 'control']:
            prob.cost.path += self._penalty_func
        elif position.lower() == 'terminal':
            prob.cost.terminal += self._penalty_func
        else:
            raise ValueError(f'position \"{position}\" is not valid')

        return prob

    def add_pre_and_post_processes(self, prob: Problem) -> Problem:
        if (self._penalty_func is None) or (self._constraint is None):
            print('Penalty method should be applied prior to adding processes')
            return prob

        _compute_penalty = lambdify(
                prob.sym_args['dynamic'], prob._substitute(self._penalty_func), use_jit_compile=prob.use_jit_compile)

        label = f'Penalty: {self._constraint.expr}'

        def _post_process(_prob: Problem, _data: Solution) -> Solution:
            _data.aux[label] = np.array(
                    [_compute_penalty(ti, xi, ui, _data.p, _data.k)
                     for ti, xi, lami, ui in zip(_data.t, _data.x.T, _data.lam.T, _data.u.T)])
            return _data

        prob.post_processes.append(_post_process)

    @staticmethod
    def _gen_sec_expr(expr: SymExpr, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> SymExpr:

        if lower_limit is None or upper_limit is None:
            raise ValueError(f'Path constraints using UTM/secant method must have lower and upper limits')

        penalty_func = regulator / sympy.cos(
            sympy.pi / 2 * (2 * expr - upper_limit - lower_limit) / (upper_limit - lower_limit)) - regulator

        return penalty_func

    @staticmethod
    def _gen_rat_expr(expr: SymExpr, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> SymExpr:

        if lower_limit is not None and upper_limit is not None:
            penalty_func = regulator \
                           * (1 / (expr - lower_limit) + 1 / (upper_limit - expr) + 4 / (lower_limit - upper_limit))
        elif lower_limit is not None:
            penalty_func = regulator / (expr - lower_limit)
        elif upper_limit is not None:
            penalty_func = regulator / (upper_limit - expr)
        else:
            raise ValueError(f'Lower or upper limit must be specified for inequality path constraint.')

        return penalty_func

    @staticmethod
    def _gen_cubic_expr(expr: SymExpr, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        penalty_func = sympy.Symbol('0')

        if lower_limit is not None:
            sign_lower = (lower_limit - expr) / ((lower_limit - expr) ** 2) ** 0.5
            penalty_func += (lower_limit - expr) ** 3 * (0.5 + 0.5 * sign_lower) / regulator

        if upper_limit is not None:
            sign_upper = (expr - upper_limit) / ((expr - upper_limit) ** 2) ** 0.5
            penalty_func += (expr - upper_limit) ** 3 * (0.5 + 0.5 * sign_upper) / regulator

        return penalty_func
