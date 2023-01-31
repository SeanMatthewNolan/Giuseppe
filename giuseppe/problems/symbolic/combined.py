from typing import Optional

from sympy import Symbol

from giuseppe.problems import SymOCP
from giuseppe.problems.symbolic.dual import SymDual
from giuseppe.utils.typing import SymMatrix


class AlgebraicControlHandler:
    def __init__(self, sym_ocp: SymOCP, sym_dual: SymDual):
        # TODO explore sympy.solveset as a replacement to 'solve'
        from sympy import solve

        self.controls = list(sym_ocp.controls)
        self.hamiltonian = sym_dual.hamiltonian

        self.dh_du = sym_dual.hamiltonian.diff(sym_ocp.controls)
        self.control_law = solve(self.dh_du, self.controls)


class DifferentialControlHandlerNumeric:
    def __init__(self, sym_ocp: SymOCP, sym_dual: SymDual):
        self.controls: list[Symbol] = list(sym_ocp.controls)

        self.h_u: SymMatrix = SymMatrix([sym_dual.hamiltonian]).jacobian(sym_ocp.controls)
        self.h_uu: SymMatrix = self.h_u.jacobian(sym_ocp.controls)
        self.h_ut: SymMatrix = self.h_u.jacobian([sym_ocp.independent])
        self.h_ux: SymMatrix = self.h_u.jacobian(sym_ocp.states)
        self.f_u: SymMatrix = sym_ocp.dynamics.jacobian(sym_ocp.controls)

        self.rhs = self.h_ut + self.h_ux @ sym_ocp.dynamics \
            + self.f_u.T @ sym_dual.costate_dynamics[:len(sym_ocp.states.flat()), :]


class DifferentialControlHandler:
    def __init__(self, sym_ocp: SymOCP, sym_dual: SymDual):
        self.controls: list[Symbol] = list(sym_ocp.controls)

        self.h_u: SymMatrix = SymMatrix([sym_dual.hamiltonian]).jacobian(sym_ocp.controls)
        self.h_uu: SymMatrix = self.h_u.jacobian(sym_ocp.controls)
        self.h_ut: SymMatrix = self.h_u.jacobian([sym_ocp.independent])
        self.h_ux: SymMatrix = self.h_u.jacobian(sym_ocp.states)
        self.f_u: SymMatrix = sym_ocp.dynamics.jacobian(sym_ocp.controls)

        self.control_dynamics = \
            -self.h_uu.LUsolve(self.h_ut + self.h_ux @ sym_ocp.dynamics
                               + self.f_u.T @ sym_dual.costate_dynamics[:len(sym_ocp.states.flat()), :])


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
