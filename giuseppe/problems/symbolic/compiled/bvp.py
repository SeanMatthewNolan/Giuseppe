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

        self.dynamics = self.compile_dynamics()
        self.boundary_conditions = self.compile_boundary_conditions()

    def compile_dynamics(self):
        _dyn = lambdify(self.sym_args, self.source_bvp.dynamics.flat(), use_jit_compile=self.use_jit_compile)

        def dynamics(
                independent: float, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:
            return np.array(_dyn(independent, states, parameters, constants))

        if self.use_jit_compile:
            dynamics = jit_compile(dynamics, self.args_numba_signature)

        return dynamics

    def compile_boundary_conditions(self):
        initial_boundary_conditions = lambdify(
                self.sym_args, tuple(self.source_bvp.boundary_conditions.initial.flat()),
                use_jit_compile=self.use_jit_compile)
        terminal_boundary_conditions = lambdify(
                self.sym_args, tuple(self.source_bvp.boundary_conditions.terminal.flat()),
                use_jit_compile=self.use_jit_compile)

        def boundary_conditions(
                independent: tuple[float, float], states: tuple[np.ndarray, np.ndarray],
                parameters: np.ndarray, constants: np.ndarray) -> np.ndarray:

            _bc_0 = np.array(initial_boundary_conditions(independent[0], states[0], parameters, constants))
            _bc_f = np.array(terminal_boundary_conditions(independent[1], states[1], parameters, constants))

            return np.concatenate((_bc_0, _bc_f))

        if self.use_jit_compile:
            boundary_conditions = jit_compile(
                    boundary_conditions,
                    (UniTuple(NumbaFloat, 2), UniTuple(NumbaArray, 2), NumbaArray, NumbaArray)
            )

        return boundary_conditions
