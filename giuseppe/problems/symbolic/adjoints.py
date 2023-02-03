from copy import deepcopy
from typing import Callable

import numpy as np

from giuseppe.problems.components.symbolic import SymCost, SymBoundaryConditions
from giuseppe.problems.protocols import Adjoints
from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.conversion import matrix_as_scalar
from giuseppe.utils.mixins import Symbolic
from giuseppe.utils.typing import SymMatrix, NumbaFloat, NumbaArray, UniTuple

from .ocp import SymOCP


class SymAdjoints(Symbolic):
    def __init__(self, ocp: SymOCP):
        Symbolic.__init__(self)

        self.source_ocp: SymOCP = deepcopy(ocp)
        self._sympify_adjoint_information(self.source_ocp)

    def _sympify_adjoint_information(self, ocp):
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

        initial_adjoint_bcs = SymMatrix([
            self.augmented_cost.initial.diff(ocp.independent) - self.hamiltonian,
            SymMatrix([self.augmented_cost.initial]).jacobian(states_and_parameters).T + self.costates
        ])
        terminal_adjoint_bcs = SymMatrix([
            self.augmented_cost.terminal.diff(ocp.independent) + self.hamiltonian,
            SymMatrix([self.augmented_cost.terminal]).jacobian(states_and_parameters).T - self.costates
        ])
        self.adjoint_boundary_conditions = SymBoundaryConditions(
                initial=initial_adjoint_bcs, terminal=terminal_adjoint_bcs
        )

        self.num_costates = len(self.costates)
        self.num_initial_adjoints = len(self.initial_adjoints)
        self.num_terminal_adjoints = len(self.terminal_adjoints)
        self.num_adjoints = len(self.adjoints)

    def compile(self, use_jit_compile: bool = True) -> 'CompAdjoints':
        return CompAdjoints(self, use_jit_compile=use_jit_compile)


class CompAdjoints(Adjoints):
    def __init__(self, source_adjoints: SymAdjoints, use_jit_compile: bool = True):
        self.use_jit_compile = use_jit_compile
        self.source_adjoints: SymAdjoints = deepcopy(source_adjoints)
        self.source_ocp: SymOCP = self.source_adjoints.source_ocp

        self._compile_adjoint_information(self.source_ocp, self.source_adjoints)

    def _compile_adjoint_information(self, sym_ocp, sym_adjoints):

        self.num_states: int = sym_ocp.num_states
        self.num_parameters: int = sym_ocp.num_parameters
        self.num_constants: int = sym_ocp.num_constants

        self.num_initial_adjoints: int = sym_adjoints.num_initial_adjoints
        self.num_terminal_adjoints: int = sym_adjoints.num_terminal_adjoints
        self.num_adjoints: int = self.num_initial_adjoints + self.num_terminal_adjoints

        self.default_values: np.ndarray = sym_ocp.default_values

        self.sym_args = {
            'dynamic': (sym_ocp.independent, sym_ocp.states.flat(),
                        sym_adjoints.costates.flat(), sym_ocp.controls.flat(),
                        sym_ocp.parameters.flat(), sym_ocp.constants.flat()),
            'static' : (sym_ocp.independent, sym_ocp.states.flat(),
                        sym_adjoints.costates.flat(), sym_ocp.controls.flat(),
                        sym_ocp.parameters.flat(), sym_adjoints.adjoints.flat(),
                        sym_ocp.constants.flat())
        }

        self.args_numba_signature = {
            'dynamic': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
            'static' : (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
        }

        self.compute_costate_dynamics = self._compile_costate_dynamics(sym_adjoints)

        _adjoint_boundary_condition_funcs = self._compile_adjoint_boundary_conditions(sym_adjoints)
        self.compute_initial_adjoint_boundary_conditions = _adjoint_boundary_condition_funcs[0]
        self.compute_terminal_adjoint_boundary_conditions = _adjoint_boundary_condition_funcs[1]
        self.compute_adjoint_boundary_conditions = _adjoint_boundary_condition_funcs[2]

        self.compute_hamiltonian = self._compile_hamiltonian(sym_adjoints)

    def _compile_costate_dynamics(self, sym_adjoints: SymAdjoints) -> Callable:
        _compute_costate_dynamics = lambdify(
                self.sym_args['dynamic'], sym_adjoints.costate_dynamics.flat(),
                use_jit_compile=self.use_jit_compile)

        def compute_costate_dynamics(
                independent: float, states: np.ndarray, controls: np.ndarray, costates: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.asarray(_compute_costate_dynamics(independent, states, controls, costates, parameters, constants))

        if self.use_jit_compile:
            compute_costate_dynamics = jit_compile(compute_costate_dynamics, self.args_numba_signature['dynamic'])

        return compute_costate_dynamics

    def _compile_adjoint_boundary_conditions(self, sym_adjoints: SymAdjoints) -> tuple:
        compute_initial_adjoint_boundary_conditions = lambdify(
                self.sym_args['static'], tuple(sym_adjoints.adjoint_boundary_conditions.initial.flat()),
                use_jit_compile=self.use_jit_compile)
        compute_terminal_adjoint_boundary_conditions = lambdify(
                self.sym_args['static'], tuple(sym_adjoints.adjoint_boundary_conditions.terminal.flat()),
                use_jit_compile=self.use_jit_compile)

        def compute_adjoint_boundary_conditions(
                independent: tuple[float, float], states: tuple[np.ndarray, np.ndarray],
                costates: tuple[np.ndarray, np.ndarray], controls: tuple[np.ndarray, np.ndarray],
                parameters: np.ndarray, adjoints: np.ndarray, constants: np.ndarray) -> np.ndarray:
            _initial_adjoint_bcs = np.asarray(compute_initial_adjoint_boundary_conditions(
                    independent[0], states[0], costates[0], controls[0], parameters, adjoints, constants))
            _terminal_adjoint_bcs = np.asarray(compute_terminal_adjoint_boundary_conditions(
                    independent[1], states[1], costates[1], controls[1], parameters, adjoints, constants))

            return np.concatenate((_initial_adjoint_bcs, _terminal_adjoint_bcs))

        if self.use_jit_compile:
            compute_adjoint_boundary_conditions = jit_compile(
                    compute_adjoint_boundary_conditions,
                    (UniTuple(NumbaFloat, 2), UniTuple(NumbaArray, 2), UniTuple(NumbaArray, 2), UniTuple(NumbaArray, 2),
                     NumbaArray, NumbaArray, NumbaArray)
            )

        return compute_initial_adjoint_boundary_conditions, compute_terminal_adjoint_boundary_conditions, \
            compute_adjoint_boundary_conditions

    def _compile_hamiltonian(self, sym_adjoints: SymAdjoints) -> Callable:
        return lambdify(self.sym_args['dynamic'], sym_adjoints.hamiltonian, use_jit_compile=self.use_jit_compile)
