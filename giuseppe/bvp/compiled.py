from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .symbolic import SymBVP
from ..utils.aliases import Array, NumbaFloat, NumbaArray
from ..utils.complilation import lambdify, jit_compile


@dataclass
class CompBoundaryConditions:
    initial: Callable
    terminal: Callable


class CompBVP:
    SUPPORTED_INPUTS = [SymBVP]

    def __init__(self, source_bvp: SymBVP):
        if type(source_bvp) not in self.SUPPORTED_INPUTS:
            raise TypeError(f'CompBVP cannot ingest type {type(source_bvp)}')

        self.src_bvp = deepcopy(source_bvp)  # source bvp is copied here for reference as it may be mutated later

        self._sym_args = (self.src_bvp.independent, self.src_bvp.states.flat(), self.src_bvp.constants.flat())
        self._args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray)

        self.dynamics = self.compile_dynamics()
        self.boundary_conditions = self.compile_boundary_conditions()

    def compile_dynamics(self):
        lam_func = lambdify(self._sym_args, tuple(self.src_bvp.dynamics.flat()))

        def dynamics(t: float, x: Array, k: Array) -> Array:
            return np.array(lam_func(t, x, k))

        return jit_compile(dynamics, signature=self._args_numba_signature)

    def compile_boundary_conditions(self):
        lam_bc0 = lambdify(self._sym_args, tuple(self.src_bvp.boundary_conditions.initial.flat()))
        lam_bcf = lambdify(self._sym_args, tuple(self.src_bvp.boundary_conditions.terminal.flat()))

        def initial_boundary_conditions(t0: float, x0: Array, k: Array) -> Array:
            return np.array(lam_bc0(t0, x0, k))

        def terminal_boundary_conditions(tf: float, xf: Array, k: Array) -> Array:
            return np.array(lam_bcf(tf, xf, k))

        return CompBoundaryConditions(
                jit_compile(initial_boundary_conditions, signature=self._args_numba_signature),
                jit_compile(terminal_boundary_conditions, signature=self._args_numba_signature),
        )

