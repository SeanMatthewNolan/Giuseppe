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


class ControlConstraintHandler(Regularizer):
    def __init__(self, regulator: Union[str, Symbol], method: str = 'atan'):
        self.regulator: Union[str, Symbol] = regulator
        self.method: str = method

        if method.lower() in ['atan', 'arctan']:
            self.expr_generator = self._gen_trig_expr
        elif method.lower() in ['trig', 'sin']:
            self.expr_generator = self._gen_trig_expr
        elif method.lower() in ['erf', 'error']:
            self.expr_generator = self._gen_trig_expr
        elif method.lower() in ['tanh']:
            self.expr_generator = self._gen_trig_expr
        elif method.lower() in ['logistic']:
            self.expr_generator = self._gen_trig_expr
        elif method.lower() in ['alg', 'algebraic']:
            self.expr_generator = self._gen_trig_expr
        else:
            raise ValueError(f'method ''{method}'' not implemented')

    # TODO: Add technique to compute real control automatically
    # TODO: Explore one-sided control functions
    def apply(self, prob: SymOCP, control_constraint: InputInequalityConstraint) -> SymOCP:

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

        return prob

    @staticmethod
    def _gen_trig_expr(pseudo_control: Symbol, lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr) \
            -> Tuple[SymExpr, SymExpr]:

        control_func = (upper_limit - lower_limit) * sympy.sin(pseudo_control) + (upper_limit + lower_limit) / 2
        error_func = regulator * sympy.cos(pseudo_control)

        return control_func, error_func

