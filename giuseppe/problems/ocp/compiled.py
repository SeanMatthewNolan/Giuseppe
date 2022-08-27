from copy import deepcopy
from typing import Union

from giuseppe.utils.compilation import lambdify
from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import NumbaFloat, NumbaArray
from .symbolic import SymOCP
from ..components.compiled import CompBoundaryConditions, CompCost


class CompOCP(Picky):
    SUPPORTED_INPUTS: type = Union[SymOCP]

    def __init__(self, source_ocp: SUPPORTED_INPUTS, use_jit_compile: bool = True):
        Picky.__init__(self, source_ocp)

        self.use_jit_compile = use_jit_compile
        self.src_ocp = deepcopy(source_ocp)  # source ocp is copied here for reference as it may be mutated later

        self.num_states = self.src_ocp.num_states
        self.num_parameters = self.src_ocp.num_parameters
        self.num_constants = self.src_ocp.num_constants
        self.num_controls = self.src_ocp.num_controls

        self.default_values = self.src_ocp.default_values

        self.sym_args = (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_ocp.controls.flat(),
                         self.src_ocp.parameters.flat(), self.src_ocp.constants.flat())

        self.args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray)

        self.dynamics = self.compile_dynamics()
        self.boundary_conditions = self.compile_boundary_conditions()
        self.cost = self.compile_cost()

    def compile_dynamics(self):
        return lambdify(self.sym_args, tuple(self.src_ocp.dynamics.flat()), use_jit_compile=self.use_jit_compile)

    def compile_boundary_conditions(self):
        initial_boundary_conditions = lambdify(self.sym_args, self.src_ocp.boundary_conditions.initial.flat(),
                                               use_jit_compile=self.use_jit_compile)
        terminal_boundary_conditions = lambdify(self.sym_args, self.src_ocp.boundary_conditions.terminal.flat(),
                                                use_jit_compile=self.use_jit_compile)

        return CompBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    def compile_cost(self):
        initial_cost = lambdify(self.sym_args, self.src_ocp.cost.initial, use_jit_compile=self.use_jit_compile)
        path_cost = lambdify(self.sym_args, self.src_ocp.cost.path, use_jit_compile=self.use_jit_compile)
        terminal_cost = lambdify(self.sym_args, self.src_ocp.cost.terminal, use_jit_compile=self.use_jit_compile)

        return CompCost(initial_cost, path_cost, terminal_cost)
