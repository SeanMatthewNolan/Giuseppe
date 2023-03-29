from typing import Optional
from copy import deepcopy

import numpy as np
# TODO explore sympy.solvest as a replacement to 'solve'
from sympy import solve, Symbol

from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.typing import SymMatrix, NumbaFloat, NumbaArray

from .dual import SymDual, CompDual


class SymAlgebraicControlHandler:
    def __init__(self, source_prob: SymDual):
        self.source_prob = deepcopy(source_prob)

        self.controls: list[Symbol] = list(self.source_prob.controls)
        self.control_functions = solve(self.source_prob.control_law, self.controls)

    def compile(self, source_comp_dual: Optional[CompDual] = None,
                use_jit_compile: bool = True) -> 'CompAlgebraicControlHandler':

        if source_comp_dual is None:
            source_comp_dual = self.source_prob.compile(use_jit_compile=use_jit_compile)

        return CompAlgebraicControlHandler(self, source_comp_dual, use_jit_compile=use_jit_compile)


class SymDifferentialControlHandler:
    def __init__(self, source_prob: SymDual):
        self.source_prob = deepcopy(source_prob)

        self.controls: list[Symbol] = list(self.source_prob.controls)

        self.h_u: SymMatrix = SymMatrix([self.source_prob.hamiltonian]).jacobian(self.source_prob.controls)
        self.h_uu: SymMatrix = self.h_u.jacobian(self.source_prob.controls)
        self.h_ut: SymMatrix = self.h_u.jacobian([self.source_prob.independent])
        self.h_ux: SymMatrix = self.h_u.jacobian(self.source_prob.states)
        self.f_u: SymMatrix = self.source_prob.dynamics.jacobian(self.source_prob.controls)

        self.control_dynamics = \
            -self.h_uu.LUsolve(self.h_ut + self.h_ux @ self.source_prob.dynamics + self.f_u.T
                               @ self.source_prob.costate_dynamics[:len(self.source_prob.states.flat()), :])

    def compile(self, source_comp_dual: Optional[CompDual] = None,
                use_jit_compile: bool = True) -> 'CompDifferentialControlHandler':

        if source_comp_dual is None:
            source_comp_dual = self.source_prob.compile(use_jit_compile=use_jit_compile)

        return CompDifferentialControlHandler(self, source_comp_dual, use_jit_compile=use_jit_compile)


class CompAlgebraicControlHandler:
    def __init__(self, source_handler: SymAlgebraicControlHandler, source_comp_dual: CompDual,
                 use_jit_compile: bool = True):
        self.source_handler = deepcopy(source_handler)
        self.source_comp_dual = deepcopy(source_comp_dual)

        self.use_jit_compile = use_jit_compile
        self.use_jit_compile = source_comp_dual.use_jit_compile
        self.sym_args = (self.source_comp_dual.source_dual.independent,
                         self.source_comp_dual.source_dual.states.flat(),
                         self.source_comp_dual.source_dual.costates.flat(),
                         self.source_comp_dual.source_dual.parameters.flat(),
                         self.source_comp_dual.source_dual.constants.flat())
        self.args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray)
        self.compute_control = self.compile_control()

    def compile_control(self):
        control_law = self.source_handler.control_functions

        if isinstance(control_law, dict):
            control_law = tuple(zip([control_law[_ui] for _ui in self.source_handler.controls]))

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
    def __init__(self, source_handler: SymDifferentialControlHandler, source_comp_dual: CompDual,
                 use_jit_compile: bool = True):
        self.source_handler = deepcopy(source_handler)
        self.source_comp_dual = deepcopy(source_comp_dual)

        self.use_jit_compile = use_jit_compile

        self.sym_args = self.source_comp_dual.sym_args['dynamic']
        self.args_numba_signature = self.source_comp_dual.args_numba_signature['dynamic']

        self.compute_control_dynamics, self.compute_h_uu = self.compile_control_dynamics()
        self.compute_control_boundary_conditions = self.source_comp_dual.compute_control_law

    def compile_control_dynamics(self):
        _compute_control_dynamics = lambdify(self.sym_args, tuple(self.source_handler.control_dynamics.flat()),
                                             use_jit_compile=self.use_jit_compile)

        _compute_h_uu = lambdify(self.sym_args, self.source_handler.h_uu, use_jit_compile=self.use_jit_compile)

        def compute_control_dynamics(
                independent: float, states: np.ndarray, controls: np.ndarray, costates: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.asarray(_compute_control_dynamics(independent, states, controls, costates, parameters, constants))

        def compute_h_uu(
                independent: float, states: np.ndarray, controls: np.ndarray, costates: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            # return np.array(_compute_h_uu(independent, states, controls, costates, parameters, constants), ndmin=2)
            return _compute_h_uu(independent, states, controls, costates, parameters, constants)

        if self.use_jit_compile:
            compute_control_dynamics = jit_compile(compute_control_dynamics, self.args_numba_signature)
            compute_h_uu = jit_compile(compute_h_uu, self.args_numba_signature)

        return compute_control_dynamics, compute_h_uu
