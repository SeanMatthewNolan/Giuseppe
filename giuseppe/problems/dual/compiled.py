from copy import deepcopy
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.utils.complilation import lambdify, jit_compile
from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import NumbaFloat, NumbaArray
from .symbolic import SymDual, SymDualOCP, SymOCP
from ..bvp.compiled import CompBoundaryConditions
from ..ocp.compiled import CompCost


class CompDual(Picky):
    SUPPORTED_INPUTS: type = Union[SymDual]

    def __init__(self, source_dual: SUPPORTED_INPUTS):
        Picky.__init__(self, source_dual)

        self.src_dual: Union[SymDual] = deepcopy(source_dual)
        self.src_ocp: Union[SymOCP] = self.src_dual.src_ocp

        self._sym_args = {
            'initial': (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_dual.costates.flat(),
                        self.src_ocp.controls.flat(), self.src_dual.initial_adjoints, self.src_ocp.constants.flat()),
            'dynamic': (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_dual.costates.flat(),
                        self.src_ocp.controls.flat(), self.src_ocp.constants.flat()),
            'terminal': (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_dual.costates.flat(),
                         self.src_ocp.controls.flat(), self.src_dual.terminal_adjoints, self.src_ocp.constants.flat()),
        }

        self._args_numba_signature = {
            'initial': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
            'dynamic': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
            'terminal': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
        }

        self.costate_dynamics = self.compile_costate_dynamics()
        self.adjoined_boundary_conditions = self.compile_adjoined_boundary_conditions()
        self.augmented_cost = self.compile_augemented_cost()
        self.hamiltonian = self.augmented_cost.path

    def compile_costate_dynamics(self):
        lam_func = lambdify(self._sym_args['dynamic'], tuple(self.src_dual.costate_dynamics.flat()))

        def costate_dynamics(t: float, x: ArrayLike, lam: ArrayLike, u: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_func(t, x, lam, u, k))

        return jit_compile(costate_dynamics, signature=self._args_numba_signature['dynamic'])

    def compile_adjoined_boundary_conditions(self):
        lam_bc0 = lambdify(self._sym_args['initial'],
                           tuple(self.src_dual.adjoined_boundary_conditions.initial.flat()))
        lam_bcf = lambdify(self._sym_args['terminal'],
                           tuple(self.src_dual.adjoined_boundary_conditions.terminal.flat()))

        def initial_boundary_conditions(t0: float, x0: ArrayLike, lam0: ArrayLike, u0: ArrayLike,
                                        nu0: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_bc0(t0, x0, lam0, u0, nu0, k))

        def terminal_boundary_conditions(tf: float, xf: ArrayLike, lamf: ArrayLike, uf: ArrayLike,
                                         nuf: ArrayLike, k: ArrayLike) -> ArrayLike:
            return np.array(lam_bcf(tf, xf, lamf, uf, nuf, k))

        return CompBoundaryConditions(
                jit_compile(initial_boundary_conditions, signature=self._args_numba_signature['initial']),
                jit_compile(terminal_boundary_conditions, signature=self._args_numba_signature['terminal']),
        )

    def compile_augemented_cost(self):
        lam_cost_0 = lambdify(self._sym_args['initial'], self.src_dual.augmented_cost.initial)
        lam_ham = lambdify(self._sym_args['dynamic'], self.src_dual.augmented_cost.path)
        lam_cost_f = lambdify(self._sym_args['terminal'], self.src_dual.augmented_cost.terminal)

        def initial_aug_cost(t0: float, x0: ArrayLike, lam0: ArrayLike, u0: ArrayLike,
                             nu0: ArrayLike, k: ArrayLike) -> float:
            return lam_cost_0(t0, x0, lam0, u0, nu0, k)

        def hamiltonian(t: float, x: ArrayLike, lam: ArrayLike, u: ArrayLike, k: ArrayLike) -> float:
            return lam_ham(t, x, lam, u, k)

        def terminal_aug_cost(tf: float, xf: ArrayLike, lamf: ArrayLike, uf: ArrayLike,
                              nuf: ArrayLike, k: ArrayLike) -> float:
            return lam_cost_f(tf, xf, lamf, uf, nuf, k)

        return CompCost(
                jit_compile(initial_aug_cost, signature=self._args_numba_signature['initial']),
                jit_compile(hamiltonian, signature=self._args_numba_signature['dynamic']),
                jit_compile(terminal_aug_cost, signature=self._args_numba_signature['terminal']),
        )


class CompDualOCP(Picky):
    SUPPORTED_INPUTS: type = Union[SymDualOCP]

    def __init__(self, source_ocp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_ocp)

        self.source_ocp = deepcopy(source_ocp)  # source ocp is copied here for reference as it may be mutated later
