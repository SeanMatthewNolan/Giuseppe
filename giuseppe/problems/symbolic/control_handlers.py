from copy import deepcopy

import numpy as np
# TODO explore sympy.solvest as a replacement to 'solve'
from sympy import solve, Symbol

from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.typing import SymMatrix, NumbaFloat, NumbaArray

from .ocp import SymOCP, CompOCP
from .dual import SymDual, CompDual


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


class CompImplicitAlgebraicControlHandler:
    def __init__(self, source_handler: ImplicitAlgebraicControlHandler, source_comp_dual: CompDual):
        self.source_handler = deepcopy(source_handler)
        self.source_comp_dual = deepcopy(source_comp_dual)

        self.use_jit_compile = source_comp_dual.use_jit_compile
        self.sym_args = (self.source_comp_dual.source_ocp.independent,
                         self.source_comp_dual.source_ocp.states.flat(),
                         self.source_comp_dual.source_dual.costates.flat(),
                         self.source_comp_dual.source_ocp.controls.flat(),
                         self.source_comp_dual.source_ocp.parameters.flat(),
                         self.source_comp_dual.source_ocp.constants.flat())
        self.args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray)
        self.compute_control_law = self.compile_control_law()

    def compile_control_law(self):
        _compute_control_law = lambdify(self.sym_args, self.source_handler.control_law,
                                        use_jit_compile=self.use_jit_compile)

        def compute_control_law(independent: float, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                                parameters: np.ndarray, constants: np.ndarray):
            return np.asarray(_compute_control_law(independent, states, costates, controls, parameters, constants))

        if self.use_jit_compile:
            compute_control_law = jit_compile(compute_control_law, self.args_numba_signature)

        return compute_control_law


class CompExplicitAlgebraicControlHandler:
    def __init__(self, source_handler: ExplicitAlgebraicControlHandler, source_comp_dual: CompDual):
        self.source_handler = deepcopy(source_handler)
        self.source_comp_dual = deepcopy(source_comp_dual)

        self.use_jit_compile = source_comp_dual.use_jit_compile
        self.sym_args = (self.source_comp_dual.source_ocp.independent,
                         self.source_comp_dual.source_ocp.states.flat(),
                         self.source_comp_dual.source_dual.costates.flat(),
                         self.source_comp_dual.source_ocp.parameters.flat(),
                         self.source_comp_dual.source_ocp.constants.flat())
        self.args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray)
        self.compute_control = self.compile_control()

    def compile_control(self):
        control_law = self.source_handler.control_law

        num_options = len(control_law)

        if num_options == 1:
            _compute_control = lambdify(self.sym_args, list(control_law[-1]), use_jit_compile=self.use_jit_compile)

            def compute_control(t: float, x: np.ndarray, lam: np.ndarray, p: np.ndarray, k: np.ndarray):
                return np.asarray(_compute_control(t, x, lam, p, k))

        else:
            hamiltonian = self.source_comp_dual.compute_hamiltonian
            _compute_control = lambdify(self.sym_args, SymMatrix(control_law), use_jit_compile=self.use_jit_compile)

            def compute_control(t: float, x: np.ndarray, lam: np.ndarray, p: np.ndarray, k: np.ndarray):
                control_options = _compute_control(t, x, lam, p, k)
                hamiltonians = np.empty((num_options,))
                for idx in range(num_options):
                    hamiltonians[idx] = hamiltonian(t, x, lam, control_options[idx], p, k)

                return control_options[np.argmin(hamiltonians)]

        if self.use_jit_compile:
            compute_control = jit_compile(compute_control, self.args_numba_signature)

        return compute_control


class CompDifferentialControlHandler:
    def __init__(self, source_handler: DifferentialControlHandler, source_comp_dual: CompDual):
        self.source_handler = deepcopy(source_handler)
        self.source_comp_dual = deepcopy(source_comp_dual)

        self.use_jit_compile = self.source_comp_dual.use_jit_compile

        self.sym_args = self.source_comp_dual.sym_args['dynamic']
        self.args_numba_signature = self.source_comp_dual.args_numba_signature['dynamic']

        self.compute_control_dynamics = self.compile_control_dynamics()
        self.compute_control_boundary_conditions = self.compile_control_boundary_conditions()

    def compile_control_dynamics(self):
        _compute_control_dynamics = lambdify(self.sym_args, self.source_handler.control_dynamics.flat(),
                                             use_jit_compile=self.use_jit_compile)

        def compute_control_dynamics(
                independent: float, states: np.ndarray, controls: np.ndarray, costates: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.array(_compute_control_dynamics(independent, states, controls, costates, parameters, constants))

        if self.use_jit_compile:
            compute_control_dynamics = jit_compile(compute_control_dynamics, self.args_numba_signature)

        return compute_control_dynamics

    def compile_control_boundary_conditions(self):
        _compute_control_boundary_conditions = lambdify(self.sym_args, self.source_handler.h_u.flat(),
                                                        use_jit_compile=self.use_jit_compile)

        def compute_control_boundary_conditions(
                independent: float, states: np.ndarray, controls: np.ndarray, costates: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.asarray(_compute_control_boundary_conditions(
                    independent, states, controls, costates, parameters, constants))

        if self.use_jit_compile:
            compute_control_boundary_conditions = jit_compile(
                    compute_control_boundary_conditions, self.args_numba_signature)

        return compute_control_boundary_conditions
