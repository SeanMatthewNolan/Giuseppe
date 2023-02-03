from copy import deepcopy
from typing import Callable

import numpy as np

from giuseppe.problems.components.symbolic import SymCost, SymBoundaryConditions
from giuseppe.problems.ocp.symbolic import SymOCP
from giuseppe.problems.protocols import Dual
from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.conversion import matrix_as_scalar
from giuseppe.utils.mixins import Symbolic
from giuseppe.utils.typing import SymMatrix, NumbaFloat, NumbaArray, UniTuple

from .ocp import SymOCP


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


class CompDual(Dual):
    def __init__(self, source_dual: SymDual, use_jit_compile: bool = True):
        self.use_jit_compile = use_jit_compile
        self.source_dual: SymDual = deepcopy(source_dual)
        self.source_ocp: SymOCP = self.source_dual.source_ocp

        self.num_states: int = self.source_ocp.num_states
        self.num_parameters: int = self.source_ocp.num_parameters
        self.num_constants: int = self.source_ocp.num_constants

        self.num_initial_adjoints: int = self.source_dual.num_initial_adjoints
        self.num_terminal_adjoints: int = self.source_dual.num_terminal_adjoints
        self.num_adjoints: int = self.num_initial_adjoints + self.num_terminal_adjoints

        self.default_values: np.ndarray = self.source_ocp.default_values

        self.sym_args = {
            'dynamic': (self.source_ocp.independent, self.source_ocp.states.flat(), self.source_dual.costates.flat(),
                        self.source_ocp.controls.flat(), self.source_ocp.parameters.flat(),
                        self.source_ocp.constants.flat()),
            'static' : (self.source_ocp.independent, self.source_ocp.states.flat(), self.source_dual.costates.flat(),
                        self.source_ocp.controls.flat(), self.source_ocp.parameters.flat(),
                        self.source_dual.adjoints.flat(), self.source_ocp.constants.flat())
        }

        self.args_numba_signature = {
            'dynamic': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
            'static' : (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
        }

        self.compute_costate_dynamics = self.compile_costate_dynamics()

        _dual_boundary_condition_funcs = self.compile_dual_boundary_conditions()
        self.compute_initial_dual_boundary_conditions = _dual_boundary_condition_funcs[0]
        self.compute_terminal_dual_boundary_conditions = _dual_boundary_condition_funcs[1]
        self.compute_dual_boundary_conditions = _dual_boundary_condition_funcs[2]

        self.compute_hamiltonian = self.compile_hamiltonian()

    def compile_costate_dynamics(self) -> Callable:
        _compute_costate_dynamics = lambdify(
                self.sym_args['dynamic'], self.source_dual.costate_dynamics.flat(),
                use_jit_compile=self.use_jit_compile)

        def compute_costate_dynamics(
                independent: float, states: np.ndarray, controls: np.ndarray, costates: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.asarray(_compute_costate_dynamics(independent, states, controls, costates, parameters, constants))

        if self.use_jit_compile:
            compute_costate_dynamics = jit_compile(compute_costate_dynamics, self.args_numba_signature['dynamic'])

        return compute_costate_dynamics

    def compile_dual_boundary_conditions(self) -> tuple:
        compute_initial_dual_boundary_conditions = lambdify(
                self.sym_args['static'], tuple(self.source_dual.dual_boundary_conditions.initial.flat()),
                use_jit_compile=self.use_jit_compile)
        compute_terminal_dual_boundary_conditions = lambdify(
                self.sym_args['static'], tuple(self.source_dual.dual_boundary_conditions.terminal.flat()),
                use_jit_compile=self.use_jit_compile)

        def compute_dual_boundary_conditions(
                independent: tuple[float, float], states: tuple[np.ndarray, np.ndarray],
                costates: tuple[np.ndarray, np.ndarray], controls: tuple[np.ndarray, np.ndarray],
                parameters: np.ndarray, adjoints: np.ndarray, constants: np.ndarray) -> np.ndarray:
            _initial_dual_bcs = np.asarray(compute_initial_dual_boundary_conditions(
                    independent[0], states[0], costates[0], controls[0], parameters, adjoints, constants))
            _terminal_dual_bcs = np.asarray(compute_terminal_dual_boundary_conditions(
                    independent[1], states[1], costates[1], controls[1], parameters, adjoints, constants))

            return np.concatenate((_initial_dual_bcs, _terminal_dual_bcs))

        if self.use_jit_compile:
            compute_dual_boundary_conditions = jit_compile(
                    compute_dual_boundary_conditions,
                    (UniTuple(NumbaFloat, 2), UniTuple(NumbaArray, 2), UniTuple(NumbaArray, 2), UniTuple(NumbaArray, 2),
                     NumbaArray, NumbaArray, NumbaArray)
            )

        return compute_initial_dual_boundary_conditions, compute_terminal_dual_boundary_conditions, \
            compute_dual_boundary_conditions

    def compile_hamiltonian(self) -> Callable:
        return lambdify(self.sym_args['dynamic'], self.source_dual.hamiltonian, use_jit_compile=self.use_jit_compile)
