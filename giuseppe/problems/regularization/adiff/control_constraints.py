from typing import Union, Tuple, TYPE_CHECKING, TypeVar
from warnings import warn

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
    def __init__(self, regulator: Union[ca.SX, ca.MX], method: str = 'atan'):
        self.regulator: Union[ca.SX, ca.MX] = regulator
        self.method: str = method
        self.dtype = type(regulator)

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
        pseudo_control = self.dtype.sym(bounded_control.str() + '_reg', 1)

        # CasADi can't check if sym is in vector, so check if substitution changed vector instead
        pseudo_controls = ca.substitute(prob.controls, bounded_control, pseudo_control)
        if ca.is_equal(prob.controls, pseudo_controls):
            raise ValueError(f'Control {bounded_control} not found to add constraint')
        if control_constraint.lower_limit is None or control_constraint.upper_limit is None:
            raise ValueError('Control constraints must have lower and upper limits')

        control_idx = None
        for control_idx, control in enumerate(ca.vertsplit(prob.controls)):
            if ca.is_equal(control, bounded_control):
                break

        control_expr, pseudo_expr, error_expr = self.expr_generator(pseudo_control, bounded_control,
                                                                    control_constraint.lower_limit,
                                                                    control_constraint.upper_limit,
                                                                    self.regulator)
        pseudo_control_expression = ca.substitute(prob.ca_pseudo2control(prob.controls, prob.constants),
                                                  bounded_control, control_expr)
        real_control_expression = ca.substitute(prob.ca_control2pseudo(prob.unregulated_controls, prob.constants),
                                                bounded_control, pseudo_expr)

        prob.controls = pseudo_controls
        prob.ca_pseudo2control = ca.Function('u', (prob.controls, prob.constants), (pseudo_control_expression,),
                                             ('u_reg', 'k'), ('u',))
        prob.ca_control2pseudo = ca.Function('u_reg', (prob.unregulated_controls, prob.constants),
                                             (real_control_expression,),
                                             ('u', 'k'), ('u_reg',))

        prob.eom = ca.substitute(prob.eom, bounded_control, control_expr)

        prob.inputConstraints.initial = ca.substitute(prob.inputConstraints.initial, bounded_control, control_expr)
        prob.inputConstraints.terminal = ca.substitute(prob.inputConstraints.terminal, bounded_control, control_expr)

        prob.inputCost.initial = ca.substitute(prob.inputCost.initial, bounded_control, control_expr)
        prob.inputCost.terminal = ca.substitute(prob.inputCost.terminal, bounded_control, control_expr)

        prob.inputCost.path = ca.substitute(prob.inputCost.path, bounded_control, control_expr) + error_expr

        if not ca.is_equal(prob.input_lower_bounds['u'][control_idx], -ca.inf)\
                or not ca.is_equal(prob.input_upper_bounds['u'][control_idx], ca.inf):
            warn(f'It is not recommended to put bounds on a regularized control! '
                 f'Consider leaving \'{bounded_control.name()}\' unbounded.')
            prob.input_lower_bounds['u'][control_idx] = prob.ca_control2pseudo(prob.input_lower_bounds['u'],
                                                                               prob.constants)[control_idx]
            prob.input_upper_bounds['u'][control_idx] = prob.ca_control2pseudo(prob.input_upper_bounds['u'],
                                                                               prob.constants)[control_idx]

        for key in prob.input_lower_bounds.keys():
            prob.input_lower_bounds[key] = ca.substitute(prob.input_lower_bounds[key], bounded_control, control_expr)
            prob.input_upper_bounds[key] = ca.substitute(prob.input_upper_bounds[key], bounded_control, control_expr)

        return prob

    @staticmethod
    def _gen_atan_expr(pseudo_control: Union[ca.SX, ca.MX], lower_limit: Union[Union[ca.SX, ca.MX], float],
                       upper_limit: Union[Union[ca.SX, ca.MX], float],
                       regulator: Union[ca.SX, ca.MX]) -> Tuple[Union[ca.SX, ca.MX], Union[ca.SX, ca.MX]]:

        control_expr = (upper_limit - lower_limit) / ca.pi * ca.atan(pseudo_control / regulator) \
                       + (upper_limit + lower_limit) / 2
        error_expr = regulator * ca.log(1 + pseudo_control**2 / regulator**2) / ca.pi

        return control_expr, error_expr

    @staticmethod
    def _gen_trig_expr(pseudo_control: Symbol, control: Symbol,
                       lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr)\
            -> Tuple[SymExpr, SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) / 2 * ca.sin(pseudo_control) + (upper_limit + lower_limit) / 2
        pseudo_expr = ca.asin((2*control - upper_limit - lower_limit) / (upper_limit - lower_limit))
        error_expr = -regulator * ca.cos(pseudo_control)

        return control_expr, pseudo_expr, error_expr

    @staticmethod
    def _gen_erf_expr(pseudo_control: Symbol, control: Symbol,
                      lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr)\
            -> Tuple[SymExpr, SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) / 2 * ca.erf(pseudo_control) + (upper_limit + lower_limit) / 2
        pseudo_expr = ca.erfinv((2*control - upper_limit - lower_limit) / (upper_limit - lower_limit))
        error_expr = regulator * (1 - ca.exp(-pseudo_control**2/regulator**2)) / ca.sqrt(ca.pi)

        return control_expr, pseudo_expr, error_expr

    @staticmethod
    def _gen_tanh_expr(pseudo_control: Symbol, control: Symbol,
                       lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr)\
            -> Tuple[SymExpr, SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) / 2 * ca.tanh(pseudo_control / regulator) \
            + (upper_limit + lower_limit) / 2
        pseudo_expr = regulator * ca.atanh((2*control - upper_limit - lower_limit) / (upper_limit - lower_limit))
        error_expr = pseudo_control * ca.tanh(pseudo_control / regulator) \
            - regulator * ca.log(ca.cosh(pseudo_control / regulator))

        return control_expr, pseudo_expr, error_expr

    @staticmethod
    def _gen_logistic_expr(pseudo_control: Symbol, control: Symbol,
                           lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr)\
            -> Tuple[SymExpr, SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) * ((1 + ca.exp(-pseudo_control / regulator))**-1 - 1 / 2) \
            + (upper_limit + lower_limit) / 2
        pseudo_expr = -regulator * ca.log(
            ((2*control - upper_limit - lower_limit) / (2*(upper_limit - lower_limit)) + 0.5)**-1 - 1)
        error_expr = - 2 * regulator * ca.log((1 + ca.exp(-pseudo_control / regulator)) / 2) \
            + 2 * pseudo_control * ((1 + ca.exp(-pseudo_control / regulator)) ** -1 - 1)

        return control_expr, pseudo_expr, error_expr

    @staticmethod
    def _gen_alg_expr(pseudo_control: Symbol, control: Symbol,
                      lower_limit: SymExpr, upper_limit: SymExpr, regulator: SymExpr)\
            -> Tuple[SymExpr, SymExpr, SymExpr]:

        control_expr = (upper_limit - lower_limit) / 2 * (pseudo_control / regulator) \
            * (1 + (pseudo_control ** 2 / regulator ** 2)) + (upper_limit + lower_limit) / 2
        pseudo_expr = ca.poly_roots(ca.poly_coeff(pseudo_control**3 + regulator**2 * pseudo_control
                                                  - regulator**3 * ((2 * control - upper_limit - lower_limit)
                                                                    / (upper_limit - lower_limit)), pseudo_control))
        error_expr = regulator * (1 - (1 + (pseudo_control ** 2 / regulator ** 2)) ** (-1/2))

        return control_expr, pseudo_expr, error_expr
