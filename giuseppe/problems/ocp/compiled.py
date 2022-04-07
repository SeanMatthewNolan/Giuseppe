from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.utils.complilation import lambdify, jit_compile
from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import NumbaFloat, NumbaArray
from .symbolic import SymOCP
from ..bvp.compiled import CompBoundaryConditions


@dataclass
class CompCost:
    initial: Callable
    path: Callable
    terminal: Callable


class CompOCP(Picky):
    SUPPORTED_INPUTS: type = Union[SymOCP]

    def __init__(self, source_ocp: SUPPORTED_INPUTS, use_jit_compile=True):
        Picky.__init__(self, source_ocp)

        self.use_jit_compile = use_jit_compile
        self.src_ocp = deepcopy(source_ocp)  # source ocp is copied here for reference as it may be mutated later

        self.num_states = len(self.src_ocp.states)
        self.num_parameters = len(self.src_ocp.parameters)
        self.num_constants = len(self.src_ocp.constants)
        self.num_controls = len(self.src_ocp.controls)

        self.default_values = self.src_ocp.default_values

        self.sym_args = {
            'static': (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_ocp.parameters.flat(),
                       self.src_ocp.constants.flat()),
            'dynamic': (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_ocp.controls.flat(),
                        self.src_ocp.parameters.flat(), self.src_ocp.constants.flat())
        }

        self.args_numba_signature = {
            'static': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray),
            'dynamic': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray)
        }

        self.dynamics = self.compile_dynamics()
        self.boundary_conditions = self.compile_boundary_conditions()
        self.cost = self.compile_cost()

    def compile_dynamics(self):
        lam_func = lambdify(self.sym_args['dynamic'], tuple(self.src_ocp.dynamics.flat()),
                            use_jit_compile=self.use_jit_compile)

        def dynamics(t: float, x: ArrayLike, u: ArrayLike, p: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_func(t, x, u, p, k))

        if self.use_jit_compile:
            return jit_compile(dynamics, signature=self.args_numba_signature['dynamic'])
        else:
            return dynamics

    def compile_boundary_conditions(self):
        lam_bc0 = lambdify(self.sym_args['static'], tuple(self.src_ocp.boundary_conditions.initial.flat()),
                           use_jit_compile=self.use_jit_compile)
        lam_bcf = lambdify(self.sym_args['static'], tuple(self.src_ocp.boundary_conditions.terminal.flat()),
                           use_jit_compile=self.use_jit_compile)

        def initial_boundary_conditions(t0: float, x0: ArrayLike, p: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_bc0(t0, x0, p, k))

        def terminal_boundary_conditions(tf: float, xf: ArrayLike, p: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_bcf(tf, xf, p, k))

        if self.use_jit_compile:
            return CompBoundaryConditions(
                    jit_compile(initial_boundary_conditions, signature=self.args_numba_signature['static']),
                    jit_compile(terminal_boundary_conditions, signature=self.args_numba_signature['static']),
            )
        else:
            return CompBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    def compile_cost(self):
        lam_cost_0 = lambdify(self.sym_args['static'], self.src_ocp.cost.initial, use_jit_compile=self.use_jit_compile)
        lam_cost_path = lambdify(self.sym_args['dynamic'], self.src_ocp.cost.path, use_jit_compile=self.use_jit_compile)
        lam_cost_f = lambdify(self.sym_args['static'], self.src_ocp.cost.terminal, use_jit_compile=self.use_jit_compile)

        def initial_cost(t0: float, x0: ArrayLike, p: ArrayLike, k: ArrayLike) -> float:
            return lam_cost_0(t0, x0, p, k)

        def path_cost(t: float, x: ArrayLike, u: ArrayLike, p: ArrayLike, k: ArrayLike) -> float:
            return lam_cost_path(t, x, u, p, k)

        def terminal_cost(tf: float, xf: ArrayLike, p: ArrayLike, k: ArrayLike) -> float:
            return lam_cost_f(tf, xf, p, k)

        if self.use_jit_compile:
            return CompCost(
                    jit_compile(initial_cost, signature=self.args_numba_signature['static']),
                    jit_compile(path_cost, signature=self.args_numba_signature['dynamic']),
                    jit_compile(terminal_cost, signature=self.args_numba_signature['static']),
            )
        else:
            return CompCost(initial_cost, path_cost, terminal_cost)
