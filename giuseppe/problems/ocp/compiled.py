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

    def __init__(self, source_ocp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_ocp)

        self.src_ocp = deepcopy(source_ocp)  # source ocp is copied here for reference as it may be mutated later

        self._sym_args = {
            'static': (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_ocp.constants.flat()),
            'dynamic': (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_ocp.controls.flat(),
                        self.src_ocp.constants.flat())
        }

        self._args_numba_signature = {
            'static': (NumbaFloat, NumbaArray, NumbaArray),
            'dynamic': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray)
        }

        self.dynamics = self.compile_dynamics()
        self.boundary_conditions = self.compile_boundary_conditions()
        self.cost = self.compile_cost()

    def compile_dynamics(self):
        lam_func = lambdify(self._sym_args['dynamic'], tuple(self.src_ocp.dynamics.flat()))

        def dynamics(t: float, x: ArrayLike, u: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_func(t, x, u, k))

        return jit_compile(dynamics, signature=self._args_numba_signature['dynamic'])

    def compile_boundary_conditions(self):
        lam_bc0 = lambdify(self._sym_args['static'], tuple(self.src_ocp.boundary_conditions.initial.flat()))
        lam_bcf = lambdify(self._sym_args['static'], tuple(self.src_ocp.boundary_conditions.terminal.flat()))

        def initial_boundary_conditions(t0: float, x0: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_bc0(t0, x0, k))

        def terminal_boundary_conditions(tf: float, xf: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_bcf(tf, xf, k))

        return CompBoundaryConditions(
                jit_compile(initial_boundary_conditions, signature=self._args_numba_signature['static']),
                jit_compile(terminal_boundary_conditions, signature=self._args_numba_signature['static']),
        )

    def compile_cost(self):
        lam_cost_0 = lambdify(self._sym_args['static'], self.src_ocp.cost.initial)
        lam_cost_path = lambdify(self._sym_args['dynamic'], self.src_ocp.cost.path)
        lam_cost_f = lambdify(self._sym_args['static'], self.src_ocp.cost.terminal)

        def initial_cost(t0: float, x0: ArrayLike, k: ArrayLike) -> float:
            return lam_cost_0(t0, x0, k)

        def path_cost(t: float, x: ArrayLike, u: ArrayLike, k: ArrayLike) -> float:
            return lam_cost_path(t, x, u, k)

        def terminal_cost(tf: float, xf: ArrayLike, k: ArrayLike) -> float:
            return lam_cost_f(tf, xf, k)

        return CompCost(
                jit_compile(initial_cost, signature=self._args_numba_signature['static']),
                jit_compile(path_cost, signature=self._args_numba_signature['dynamic']),
                jit_compile(terminal_cost, signature=self._args_numba_signature['static']),
        )
