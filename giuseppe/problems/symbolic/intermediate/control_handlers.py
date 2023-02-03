from copy import deepcopy

# TODO explore sympy.solvest as a replacement to 'solve'
from sympy import solve, Symbol

from giuseppe.utils.typing import SymMatrix

from . import SymOCP, SymDual


class ImplicitAlgebraicControlHandler:
    def __init__(self, primal: SymOCP, dual: SymDual):
        self.primal, self.dual = deepcopy(primal), deepcopy(dual)

        self.controls = primal.controls
        self.hamiltonian = dual.hamiltonian

        self.control_law = dual.hamiltonian.diff(primal.controls)


class ExplicitAlgebraicControlHandler:
    def __init__(self, primal: SymOCP, dual: SymDual):
        self.primal, self.dual = deepcopy(primal), deepcopy(dual)

        self.controls: list[Symbol] = list(primal.controls)
        self.hamiltonian = dual.hamiltonian

        self.dh_du = dual.hamiltonian.diff(primal.controls)
        self.control_law = solve(self.dh_du, self.controls)


class DifferentialControlHandler:
    def __init__(self, primal: SymOCP, dual: SymDual):
        self.primal, self.dual = deepcopy(primal), deepcopy(dual)

        self.controls: list[Symbol] = list(primal.controls)

        self.h_u: SymMatrix = SymMatrix([dual.hamiltonian]).jacobian(primal.controls)
        self.h_uu: SymMatrix = self.h_u.jacobian(primal.controls)
        self.h_ut: SymMatrix = self.h_u.jacobian([primal.independent])
        self.h_ux: SymMatrix = self.h_u.jacobian(primal.states)
        self.f_u: SymMatrix = primal.dynamics.jacobian(primal.controls)

        self.control_dynamics = \
            -self.h_uu.LUsolve(self.h_ut + self.h_ux @ primal.dynamics
                               + self.f_u.T @ dual.costate_dynamics[:len(primal.states.flat()), :])
