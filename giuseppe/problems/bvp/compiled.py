from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.utils.complilation import lambdify, jit_compile
from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import NumbaFloat, NumbaArray
from .symbolic import SymBVP


@dataclass
class CompBoundaryConditions:
    initial: Callable
    terminal: Callable


class CompBVP(Picky):
    SUPPORTED_INPUTS: type = Union[SymBVP]

    def __init__(self, source_bvp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_bvp)

        self.src_bvp = deepcopy(source_bvp)  # source dual_ocp is copied here for reference as it may be mutated later

        self.num_states = len(self.src_bvp.states)
        self.num_parameters = len(self.src_bvp.parameters)
        self.num_constants = len(self.src_bvp.constants)

        self.sym_args = (self.src_bvp.independent, self.src_bvp.states.flat(), self.src_bvp.parameters.flat(),
                         self.src_bvp.constants.flat())
        self.args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray)

        self.dynamics = self.compile_dynamics()
        self.boundary_conditions = self.compile_boundary_conditions()

    def compile_dynamics(self):
        lam_func = lambdify(self.sym_args, tuple(self.src_bvp.dynamics.flat()))

        def dynamics(t: float, x: ArrayLike, p: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_func(t, x, p, k))

        return jit_compile(dynamics, signature=self.args_numba_signature)

    def compile_boundary_conditions(self):
        lam_bc0 = lambdify(self.sym_args, tuple(self.src_bvp.boundary_conditions.initial.flat()))
        lam_bcf = lambdify(self.sym_args, tuple(self.src_bvp.boundary_conditions.terminal.flat()))

        def initial_boundary_conditions(t0: float, x0: ArrayLike, p: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_bc0(t0, x0, p, k))

        def terminal_boundary_conditions(tf: float, xf: ArrayLike, p: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_bcf(tf, xf, p, k))

        return CompBoundaryConditions(
                jit_compile(initial_boundary_conditions, signature=self.args_numba_signature),
                jit_compile(terminal_boundary_conditions, signature=self.args_numba_signature),
        )
