from giuseppe.ocp.symbolic import SymOCP, SymBoundaryConditions, SymCost
from giuseppe.utils.aliases import Symbol, SymMatrix
from giuseppe.utils.conversion import matrix_as_scalar
from giuseppe.utils.mixins import Symbolic


class SymDual(Symbolic):
    def __init__(self, ocp: SymOCP):
        super().__init__()

        self.costates = SymMatrix([self.new_sym(f'_lam_{state}') for state in ocp.states])

        self.initial_adjoints = SymMatrix(
                [self.new_sym(f'_nu_0_{idx}') for idx, _ in enumerate(ocp.boundary_conditions.initial)])
        self.terminal_adjoints = SymMatrix(
                [self.new_sym(f'_nu_f_{idx}') for idx, _ in enumerate(ocp.boundary_conditions.terminal)])

        self.hamiltonian = ocp.cost.path + matrix_as_scalar(self.costates.T @ ocp.dynamics)

        self.costate_dynamics = -self.hamiltonian.diff(ocp.states)

        self.augmented_cost = SymCost(
                ocp.cost.initial + matrix_as_scalar(self.initial_adjoints.T @ ocp.boundary_conditions.initial),
                self.hamiltonian,
                ocp.cost.terminal + matrix_as_scalar(self.terminal_adjoints.T @ ocp.boundary_conditions.terminal),
        )

        initial_adjoined_bcs = SymMatrix([
            self.augmented_cost.initial.diff(ocp.independent) - self.hamiltonian,
            SymMatrix([self.augmented_cost.initial]).jacobian(ocp.states).T + self.costates
        ])
        terminal_adjoined_bcs = SymMatrix([
            self.augmented_cost.terminal.diff(ocp.independent) + self.hamiltonian,
            SymMatrix([self.augmented_cost.terminal]).jacobian(ocp.states).T - self.costates
        ])
        self.adjoined_boundary_conditions = SymBoundaryConditions(
                initial=initial_adjoined_bcs, terminal=terminal_adjoined_bcs
        )


class AlgebraicControlHandler:
    def __init__(self, sym_ocp: SymOCP, sym_dual: SymDual):
        # TODO explore sympy.solveset as a replacement to 'solve'
        from sympy import solve

        self.controls = list(sym_ocp.controls)
        self.hamiltonian = sym_dual.hamiltonian

        self.dh_du = sym_dual.hamiltonian.diff(sym_ocp.controls)
        self.control_law = solve(self.dh_du, self.controls)


class DifferentialControlHandler:
    def __init__(self, sym_ocp: SymOCP, sym_dual: SymDual):
        self.controls: list[Symbol] = list(sym_ocp.controls)

        self.h_u: SymMatrix = SymMatrix([sym_dual.hamiltonian]).jacobian(sym_ocp.controls)
        self.h_uu: SymMatrix = self.h_u.jacobian(sym_ocp.controls)
        self.h_ut: SymMatrix = self.h_u.jacobian([sym_ocp.independent])
        self.h_ux: SymMatrix = self.h_u.jacobian(sym_ocp.states)
        self.f_u: SymMatrix = sym_ocp.dynamics.jacobian(sym_ocp.controls)

        self.control_dynamics = \
            -self.h_uu.inv() * (self.h_ut + self.h_ux @ sym_ocp.dynamics + self.f_u.T @ sym_dual.costate_dynamics)


class SymDualOCP:
    def __init__(self, sym_ocp: SymOCP, sym_dual: SymDual, control_method: str = 'differential'):
        self.ocp: SymOCP = sym_ocp
        self.dual: SymDual = sym_dual

        if control_method.lower() == 'algebraic':
            self.control_handler = AlgebraicControlHandler(sym_ocp, sym_dual)
        elif control_method.lower() == 'differential':
            self.control_handler = DifferentialControlHandler(sym_ocp, sym_dual)
        else:
            raise NotImplementedError(
                    f'\"{control_method}\" is not an implemented control method. Try \"differential\".')


