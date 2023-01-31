from typing import Optional

from sympy import Symbol

from giuseppe.problems.symbolic.intermediate.ocp import SymOCP
from giuseppe.problems.symbolic.intermediate.dual import SymDual
from giuseppe.utils.typing import SymMatrix


class AlgebraicControlHandler:
    def __init__(self, primal: SymOCP, dual: SymDual):
        # TODO explore sympy.solveset as a replacement to 'solve'
        from sympy import solve

        self.controls = list(primal.controls)
        self.hamiltonian = dual.hamiltonian

        self.dh_du = dual.hamiltonian.diff(primal.controls)
        self.control_law = solve(self.dh_du, self.controls)


class DifferentialControlHandler:
    def __init__(self, primal: SymOCP, dual: SymDual):
        self.controls: list[Symbol] = list(primal.controls)

        self.h_u: SymMatrix = SymMatrix([dual.hamiltonian]).jacobian(primal.controls)
        self.h_uu: SymMatrix = self.h_u.jacobian(primal.controls)
        self.h_ut: SymMatrix = self.h_u.jacobian([primal.independent])
        self.h_ux: SymMatrix = self.h_u.jacobian(primal.states)
        self.f_u: SymMatrix = primal.dynamics.jacobian(primal.controls)

        self.control_dynamics = \
            -self.h_uu.LUsolve(self.h_ut + self.h_ux @ primal.dynamics
                               + self.f_u.T @ dual.costate_dynamics[:len(primal.states.flat()), :])


class DifferentialControlHandlerNumeric:
    def __init__(self, primal: SymOCP, dual: SymDual):
        self.controls: list[Symbol] = list(primal.controls)

        self.h_u: SymMatrix = SymMatrix([dual.hamiltonian]).jacobian(primal.controls)
        self.h_uu: SymMatrix = self.h_u.jacobian(primal.controls)
        self.h_ut: SymMatrix = self.h_u.jacobian([primal.independent])
        self.h_ux: SymMatrix = self.h_u.jacobian(primal.states)
        self.f_u: SymMatrix = primal.dynamics.jacobian(primal.controls)

        self.rhs = self.h_ut + self.h_ux @ primal.dynamics \
            + self.f_u.T @ dual.costate_dynamics[:len(primal.states.flat()), :]


# TODO: Consider exposing OCP and Dual attributes
class SymCombined:
    def __init__(self, primal: SymOCP, dual: Optional[SymDual] = None, control_method: str = 'differential'):

        self.primal: SymOCP = primal

        if dual is None:
            self.dual: SymDual = SymDual(primal)
        else:
            self.dual: SymDual = dual

        self.control_method = control_method
        if self.control_method.lower() == 'algebraic':
            self.control_handler = AlgebraicControlHandler(primal, dual)
        elif self.control_method.lower() == 'differential':
            self.control_handler = DifferentialControlHandler(primal, dual)
        elif self.control_method.lower() == 'differential_numeric':
            self.control_handler = DifferentialControlHandlerNumeric(primal, dual)
        else:
            raise NotImplementedError(
                    f'\"{control_method}\" is not an implemented control method. Try \"differential\".')
