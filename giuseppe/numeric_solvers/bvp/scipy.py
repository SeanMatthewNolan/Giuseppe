from typing import Union, Callable

import numpy as np
from scipy.integrate import solve_bvp

from ...bvp import CompBVP, BVPSol
from ...dualization import CompDualOCP
from ...utils.complilation import jit_compile
from ...utils.mixins import Picky
from ...utils.typing import NumbaArray, NumbaMatrix, NPArray

_dyn_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_bc_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_preprocess_type = Callable[[BVPSol], tuple[NPArray, NPArray, NPArray]]
_postprocess_type = Callable[[NPArray, NPArray, NPArray, NPArray], BVPSol]


class ScipySolveBVP(Picky):
    SUPPORTED_INPUTS = Union[CompBVP, CompDualOCP]

    def __init__(self, bvp: SUPPORTED_INPUTS, do_jit_compile: bool = True,
                 tol: float = 0.001, bc_tol: float = 0.001, max_nodes: int = 1000, verbose: bool = False):
        Picky.__init__(self, bvp)

        # Options directly for scipy.solve_bvp
        self.tol: float = tol
        self.bc_tol: float = bc_tol
        self.max_nodes = max_nodes
        self.verbose = verbose

        self.do_jit_compile: bool = do_jit_compile

        dynamics, boundary_conditions, preprocess, postprocess = self.load_problem(bvp)
        self.dynamics: _dyn_type = dynamics
        self.boundary_conditions: _bc_type = boundary_conditions
        self.preprocess: _preprocess_type = preprocess
        self.postprocess: _postprocess_type = postprocess

    def load_problem(self, bvp: SUPPORTED_INPUTS) -> tuple[_dyn_type, _bc_type, _preprocess_type, _postprocess_type]:
        if isinstance(bvp, CompBVP):
            dynamics = self._generate_bvp_dynamics(bvp)
            boundary_conditions = self._generate_bvp_bcs(bvp)
            preprocess = self._preprocess_bvp_sol
            postprocess = self._postprocess_bvp_sol
        elif isinstance(bvp, CompDualOCP):
            raise NotImplementedError
        else:
            raise TypeError

        return dynamics, boundary_conditions, preprocess, postprocess

    def _generate_bvp_dynamics(self, bvp: CompBVP) -> _dyn_type:
        bvp_dyn = bvp.dynamics

        def dynamics(tau_vec: NPArray, x_vec: NPArray, p: NPArray, k: NPArray) -> NPArray:
            t0, tf = p[-2], p[-1]
            tau_mult = (tf - t0)
            t_vec = tau_vec * tau_mult + (tf + t0) / 2

            x_dot = np.empty_like(x_vec)  # Need to pre-allocate for Numba
            for idx, (ti, xi) in enumerate(zip(t_vec, x_vec.T)):
                x_dot[:, idx] = bvp_dyn(ti, xi, k)

            return x_dot * tau_mult

        if self.do_jit_compile:
            dynamics = jit_compile(dynamics, (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray))

        return dynamics

    def _generate_bvp_bcs(self, bvp: CompBVP) -> _bc_type:
        bvp_bc0 = bvp.boundary_conditions.initial
        bvp_bcf = bvp.boundary_conditions.terminal

        def boundary_conditions(x0: NPArray, xf: NPArray, p: NPArray, k: NPArray):
            t0, tf = p[-2], p[-1]
            return np.concatenate((bvp_bc0(t0, x0, k), bvp_bcf(tf, xf, k)))

        if self.do_jit_compile:
            boundary_conditions = jit_compile(boundary_conditions, (NumbaArray, NumbaArray, NumbaArray, NumbaArray))

        return boundary_conditions

    @staticmethod
    def _preprocess_bvp_sol(sol: BVPSol) -> tuple[NPArray, NPArray, NPArray]:
        t0, tf = sol.t[0], sol.t[-1]
        p_guess = np.array([t0, tf])
        tau_guess = (sol.t - (t0 + tf)/2) / (tf - t0)
        return tau_guess, sol.x, p_guess

    @staticmethod
    def _postprocess_bvp_sol(tau: NPArray, x: NPArray, p: NPArray, k: NPArray) -> BVPSol:
        t0, tf = p[-2], p[-1]
        t = (tf - t0) * tau + (t0 + tf)/2
        p = p[:-2]
        return BVPSol(t=t, x=x, p=p, k=k, converged=True)

    def solve(self, guess: Union[BVPSol], constants: NPArray) -> Union[BVPSol]:

        tau_guess, x_guess, p_guess = self.preprocess(guess)
        k = constants

        sol = solve_bvp(
                lambda tau, x, p: self.dynamics(tau, x, p, k),
                lambda x0, xf, p: self.boundary_conditions(x0, xf, p, k),
                tau_guess, x_guess, p_guess,
                tol=self.tol, bc_tol=self.bc_tol, max_nodes=self.max_nodes, verbose=self.verbose
        )

        # noinspection PyUnresolvedReferences
        if sol.success:
            # noinspection PyUnresolvedReferences
            return self.postprocess(sol.x, sol.y, sol.p, k)
        else:
            # TODO make custom exception to be handled by continuation handler
            raise RuntimeError(f'BVP solver did not converge')
