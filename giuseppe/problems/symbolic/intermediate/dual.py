from copy import deepcopy

from giuseppe.problems.components.symbolic import SymCost, SymBoundaryConditions
from giuseppe.problems.ocp.symbolic import SymOCP
from giuseppe.utils.conversion import matrix_as_scalar
from giuseppe.utils.mixins import Symbolic
from giuseppe.utils.typing import SymMatrix


class SymDual(Symbolic):
    def __init__(self, ocp: SymOCP):
        Symbolic.__init__(self)

        self.source_ocp: SymOCP = deepcopy(ocp)

        states_and_parameters = SymMatrix(ocp.states.flat() + ocp.parameters.flat())

        self.costates = SymMatrix([self.new_sym(f'_lam_{state}') for state in states_and_parameters])

        self.initial_adjoints = SymMatrix(
                [self.new_sym(f'_nu_0_{idx}') for idx, _ in enumerate(ocp.boundary_conditions.initial)])
        self.terminal_adjoints = SymMatrix(
                [self.new_sym(f'_nu_f_{idx}') for idx, _ in enumerate(ocp.boundary_conditions.terminal)])
        self.adjoints = self.initial_adjoints.col_join(self.terminal_adjoints)

        self.hamiltonian = ocp.cost.path + matrix_as_scalar(self.costates[:len(ocp.states.flat()), :].T @ ocp.dynamics)

        self.costate_dynamics = -self.hamiltonian.diff(states_and_parameters)

        self.augmented_cost = SymCost(
                ocp.cost.initial + matrix_as_scalar(self.initial_adjoints.T @ ocp.boundary_conditions.initial),
                self.hamiltonian,
                ocp.cost.terminal + matrix_as_scalar(self.terminal_adjoints.T @ ocp.boundary_conditions.terminal),
        )

        initial_dual_bcs = SymMatrix([
            self.augmented_cost.initial.diff(ocp.independent) - self.hamiltonian,
            SymMatrix([self.augmented_cost.initial]).jacobian(states_and_parameters).T + self.costates
        ])
        terminal_dual_bcs = SymMatrix([
            self.augmented_cost.terminal.diff(ocp.independent) + self.hamiltonian,
            SymMatrix([self.augmented_cost.terminal]).jacobian(states_and_parameters).T - self.costates
        ])
        self.dual_boundary_conditions = SymBoundaryConditions(
                initial=initial_dual_bcs, terminal=terminal_dual_bcs
        )

        self.num_costates = len(self.costates)
        self.num_initial_adjoints = len(self.initial_adjoints)
        self.num_terminal_adjoints = len(self.terminal_adjoints)
        self.num_adjoints = len(self.adjoints)
