from giuseppe.ocp.symbolic import SymOCP, SymBoundaryConditions, SymCost
from giuseppe.utils.mixins import Symbolic
from giuseppe.utils.aliases import SymMatrix
from giuseppe.utils.functions import as_scalar


class SymDual(Symbolic):
    def __init__(self, ocp: SymOCP):
        super().__init__()

        self.costates = SymMatrix([self.new_sym(f'_lam_{state}') for state in ocp.states])

        self.initial_adjoints = SymMatrix(
                [self.new_sym(f'_nu_0_{idx}') for idx, _ in enumerate(ocp.boundary_conditions.initial)])
        self.terminal_adjoints = SymMatrix(
                [self.new_sym(f'_nu_f_{idx}') for idx, _ in enumerate(ocp.boundary_conditions.terminal)])

        self.hamiltonian = ocp.cost.path + as_scalar(self.costates.T @ ocp.dynamics)

        self.augmented_cost = SymCost(
                ocp.cost.initial + as_scalar(self.initial_adjoints.T @ ocp.boundary_conditions.initial),
                self.hamiltonian,
                ocp.cost.terminal + as_scalar(self.terminal_adjoints.T @ ocp.boundary_conditions.terminal),
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
