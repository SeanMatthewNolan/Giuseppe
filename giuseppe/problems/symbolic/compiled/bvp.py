from copy import deepcopy

import numpy as np

from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.typing import NumbaFloat, NumbaArray, UniTuple
from ...protocols import BVP
from ..intermediate import SymBVP


class CompBVP(BVP):
    def __init__(self, source_bvp: SymBVP, use_jit_compile=True):
        self.use_jit_compile = use_jit_compile
        self.source_bvp = deepcopy(source_bvp)

        self.num_states = len(self.source_bvp.states)
        self.num_parameters = len(self.source_bvp.parameters)
        self.num_constants = len(self.source_bvp.constants)
        self.default_values = self.source_bvp.default_values

        self.sym_args = (self.source_bvp.independent, self.source_bvp.states.flat(), self.source_bvp.parameters.flat(),
                         self.source_bvp.constants.flat())
        self.args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray)

        self.compute_dynamics = self.compile_dynamics()

        _boundary_condition_funcs = self.compile_boundary_conditions()
        self.compute_initial_boundary_conditions = _boundary_condition_funcs[0]
        self.compute_terminal_boundary_conditions = _boundary_condition_funcs[1]
        self.compute_boundary_conditions = _boundary_condition_funcs[2]

    def compile_dynamics(self):
        _compute_dynamics = lambdify(self.sym_args, self.source_bvp.dynamics.flat(), use_jit_compile=self.use_jit_compile)

        def compute_dynamics(
                independent: float, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.array(_compute_dynamics(independent, states, parameters, constants))

        if self.use_jit_compile:
            compute_dynamics = jit_compile(compute_dynamics, self.args_numba_signature)

        return compute_dynamics

    def compile_boundary_conditions(self):
        compute_initial_boundary_conditions = lambdify(
                self.sym_args, tuple(self.source_bvp.boundary_conditions.initial.flat()),
                use_jit_compile=self.use_jit_compile)
        compute_terminal_boundary_conditions = lambdify(
                self.sym_args, tuple(self.source_bvp.boundary_conditions.terminal.flat()),
                use_jit_compile=self.use_jit_compile)

        def compute_boundary_conditions(
                independent: tuple[float, float], states: tuple[np.ndarray, np.ndarray],
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:

            _bc_0 = np.array(compute_initial_boundary_conditions(independent[0], states[0], parameters, constants))
            _bc_f = np.array(compute_terminal_boundary_conditions(independent[1], states[1], parameters, constants))

            return np.concatenate((_bc_0, _bc_f))

        if self.use_jit_compile:
            compute_boundary_conditions = jit_compile(
                    compute_boundary_conditions,
                    (UniTuple(NumbaFloat, 2), UniTuple(NumbaArray, 2), NumbaArray, NumbaArray)
            )

        return compute_initial_boundary_conditions, compute_terminal_boundary_conditions, compute_boundary_conditions
