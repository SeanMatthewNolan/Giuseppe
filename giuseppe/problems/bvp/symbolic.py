from itertools import permutations
from typing import Optional, Union

import numpy as np
from sympy import Symbol, topological_sort

from giuseppe.problems.bvp.input import InputBVP
from giuseppe.problems.components.input import InputInequalityConstraints
from giuseppe.problems.components.symbolic import SymBoundaryConditions, SymNamedExpr
from giuseppe.utils.mixins import Symbolic
from giuseppe.utils.typing import SymMatrix, EMPTY_SYM_MATRIX, SYM_NULL, SymExpr


class SymBVP(Symbolic):
    def __init__(self, input_data: Optional[InputBVP] = None):
        super().__init__()

        self.independent: Symbol = SYM_NULL
        self.states: SymMatrix = EMPTY_SYM_MATRIX
        self.parameters: SymMatrix = EMPTY_SYM_MATRIX
        self.dynamics: SymMatrix = EMPTY_SYM_MATRIX
        self.constants: SymMatrix = EMPTY_SYM_MATRIX

        # Holding on to these for future post-processing
        self.expressions: list[SymNamedExpr] = []

        self.boundary_conditions = SymBoundaryConditions()

        self.default_values = np.array([])

        if isinstance(input_data, InputBVP):
            self.process_data_from_input(input_data)

    def process_variables_from_input(self, input_data: InputBVP):
        self.independent = self.new_sym(input_data.independent)
        self.states = SymMatrix([self.new_sym(state.name) for state in input_data.states])
        self.parameters = SymMatrix([self.new_sym(parameter) for parameter in input_data.parameters])
        self.constants = SymMatrix([self.new_sym(constant.name) for constant in input_data.constants])

        self.default_values = np.array([constant.default_value for constant in input_data.constants])

        self.expressions = [SymNamedExpr(self.new_sym(expr.name)) for expr in input_data.expressions]

    def process_expr_from_input(self, input_data: InputBVP):
        self.dynamics = SymMatrix([self.sympify(state.eom) for state in input_data.states])
        self.boundary_conditions.initial = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.initial])
        self.boundary_conditions.terminal = SymMatrix(
                [self.sympify(constraint) for constraint in input_data.constraints.terminal])

        for sym_expr, in_expr in zip(self.expressions, input_data.expressions):
            sym_expr.expr = self.sympify(in_expr.expr)

        self.order_named_expressions()

    def order_named_expressions(self):
        interdependencies = []
        for expr_i, expr_j in permutations(self.expressions, 2):
            if expr_i.expr.has(expr_j.sym):
                interdependencies.append((expr_i, expr_j))

        self.expressions = topological_sort((self.expressions, interdependencies), key=lambda _expr: str(_expr.sym))

    def process_inequality_constraints(self, input_inequality_constraints: InputInequalityConstraints):
        # TODO Evaluate symbolically before
        for position in ['initial', 'path', 'terminal', 'control']:
            for constraint in input_inequality_constraints.__getattribute__(position):
                if constraint.regularizer is None:
                    raise NotImplementedError('Inequality constraint without regularizer not yet implemented')
                else:
                    constraint.regularizer.apply(self, constraint, position)

    def substitute(self, sym_expr: Union[SymExpr, SymMatrix]):
        sub_pairs = [(named_expr.sym, named_expr.expr) for named_expr in self.expressions]
        return sym_expr.subs(sub_pairs)

    def perform_substitutions(self):
        self.dynamics = self.substitute(self.dynamics)
        self.boundary_conditions.initial = self.substitute(self.boundary_conditions.initial)
        self.boundary_conditions.terminal = self.substitute(self.boundary_conditions.terminal)

    def process_data_from_input(self, input_data: InputBVP):
        self.process_variables_from_input(input_data)
        self.process_expr_from_input(input_data)
        self.process_inequality_constraints(input_data.inequality_constraints)
        self.perform_substitutions()
