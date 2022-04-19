from copy import deepcopy
from typing import Union

from giuseppe.utils.complilation import lambdify
from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import NumbaFloat, NumbaArray
from .symbolic import SymBVP
from ..components.compiled import CompBoundaryConditions


class CompBVP(Picky):
    SUPPORTED_INPUTS: type = Union[SymBVP]

    def __init__(self, source_bvp: SUPPORTED_INPUTS, use_jit_compile=True):
        Picky.__init__(self, source_bvp)

        self.use_jit_compile = use_jit_compile
        self.src_bvp = deepcopy(source_bvp)  # source dual_ocp is copied here for reference as it may be mutated later

        self.num_states = len(self.src_bvp.states)
        self.num_parameters = len(self.src_bvp.parameters)
        self.num_constants = len(self.src_bvp.constants)
        self.default_values = self.src_bvp.default_values

        self.sym_args = (self.src_bvp.independent, self.src_bvp.states.flat(), self.src_bvp.parameters.flat(),
                         self.src_bvp.constants.flat())
        self.args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray)

        self.dynamics = self.compile_dynamics()
        self.boundary_conditions = self.compile_boundary_conditions()

    def compile_dynamics(self):
        return lambdify(self.sym_args, self.src_bvp.dynamics.flat(), use_jit_compile=self.use_jit_compile)

    def compile_boundary_conditions(self):
        initial_boundary_conditions = lambdify(self.sym_args, tuple(self.src_bvp.boundary_conditions.initial.flat()),
                                               use_jit_compile=self.use_jit_compile)
        terminal_boundary_conditions = lambdify(self.sym_args, tuple(self.src_bvp.boundary_conditions.terminal.flat()),
                                                use_jit_compile=self.use_jit_compile)

        return CompBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)
