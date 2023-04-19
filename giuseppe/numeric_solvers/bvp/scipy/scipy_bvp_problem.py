from __future__ import annotations

from copy import deepcopy
from typing import Union

from giuseppe.utils.compilation import jit_compile, check_if_can_jit_compile
from giuseppe.utils.typing import NumbaArray, NumbaMatrix

import numpy as np

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import BVP, VectorizedBVP

from .scipy_types import _scipy_bvp_sol, _dyn_type, _bc_type


class SciPyBVP:
    def __init__(self, source_bvp: Union[BVP, VectorizedBVP], use_jit_compile: bool = True):
        self.source_bvp = deepcopy(source_bvp)
        self.use_jit_compile = check_if_can_jit_compile(use_jit_compile, self.source_bvp)

        self.compute_dynamics = self._compile_dynamics()
        self.compute_boundary_conditions = self._compile_boundary_conditions()

        if self.use_jit_compile:
            self.compute_dynamics = jit_compile(
                    self.compute_dynamics, (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray))
            self.compute_boundary_conditions = jit_compile(
                    self.compute_boundary_conditions, (NumbaArray, NumbaArray, NumbaArray, NumbaArray))

        self.annotations = source_bvp.annotations

    def _compile_dynamics(self) -> _dyn_type:

        if hasattr(self.source_bvp, 'compute_dynamics_vectorized'):
            bvp_dyn = self.source_bvp.compute_dynamics
            vec_bvp_dyn = self.source_bvp.compute_dynamics_vectorized

            def compute_dynamics(tau_vec: np.ndarray, x_vec: np.ndarray, p: np.ndarray, k: np.ndarray) -> np.ndarray:
                t0, tf = p[-2], p[-1]
                # _p = p[:-2]
                tau_mult = (tf - t0)
                t_vec = tau_vec * tau_mult + t0

                x_dot = vec_bvp_dyn(t_vec, x_vec, p[:-2], k)

                return x_dot * tau_mult

            return compute_dynamics

        else:
            bvp_dyn = self.source_bvp.compute_dynamics

            def compute_dynamics(tau_vec: np.ndarray, x_vec: np.ndarray, p: np.ndarray, k: np.ndarray) -> np.ndarray:
                t0, tf = p[-2], p[-1]
                # _p = p[:-2]
                tau_mult = (tf - t0)
                t_vec = tau_vec * tau_mult + t0

                x_dot = np.empty_like(x_vec)  # Need to pre-allocate for Numba
                for idx, (ti, xi) in enumerate(zip(t_vec, x_vec.T)):
                    x_dot[:, idx] = bvp_dyn(ti, xi, p[:-2], k)

                return x_dot * tau_mult

            return compute_dynamics

    def _compile_boundary_conditions(self) -> _bc_type:
        _bvp_compute_initial_boundary_conditions = self.source_bvp.compute_initial_boundary_conditions
        _bvp_compute_terminal_boundary_conditions = self.source_bvp.compute_terminal_boundary_conditions

        def boundary_conditions(x0: np.ndarray, xf: np.ndarray, p: np.ndarray, k: np.ndarray):
            # _t = p[-2:]
            # _p = p[:-2]
            return np.concatenate((
                _bvp_compute_initial_boundary_conditions(p[-2], x0, p[:-2], k),
                _bvp_compute_terminal_boundary_conditions(p[-1], xf, p[:-2], k),
            ))

        return boundary_conditions

    def preprocess(self, guess: Solution) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        guess = self.source_bvp.preprocess_data(guess)
        t0, tf = guess.t[0], guess.t[-1]
        p_guess = np.concatenate((guess.p, np.array([t0, tf])))
        tau_guess = (guess.t - t0) / (tf - t0)

        return tau_guess, guess.x, p_guess

    def post_process(self, scipy_sol: _scipy_bvp_sol, constants) -> Solution:
        tau: np.ndarray = scipy_sol.x
        x: np.ndarray = scipy_sol.y
        p: np.ndarray = scipy_sol.p

        t0, tf = p[-2], p[-1]
        t = (tf - t0) * tau + t0
        p = p[:-2]

        if scipy_sol.status == 0:
            aux = {}
        else:
            aux = {'solver status': scipy_sol.status}

        return self.source_bvp.post_process_data(
                Solution(
                    t=t, x=x, p=p, k=constants,
                    converged=scipy_sol.success,
                    annotations=self.annotations,
                    aux=aux
                )
        )
