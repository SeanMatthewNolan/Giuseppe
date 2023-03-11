"""
This solver follows the algorithms given in

"Parallel collocation solution of index-1 BVP-DAEs arising from constrained optimal control problems" - Fabien

and

"A BVP Solver Based on Residual Control and the MATLAB PSE

"""

from typing import Optional, Tuple
from copy import deepcopy

import numpy as np

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import Dual, VectorizedDual
from giuseppe.problems.conversions import vectorize

from ...protocol import NumericSolver


class DualCollocation(NumericSolver):
    def __init__(self, prob: Dual):
        if not isinstance(prob, VectorizedDual):
            prob: VectorizedDual = vectorize(prob)

        self.prob: VectorizedDual = deepcopy(prob)

        self._compute_x_dot = self.prob.compute_dynamics_vectorized
        self._compute_lam_dot = self.prob.compute_costate_dynamics_vectorized

    def solve(self, guess: Solution, constants: Optional[np.ndarray] = None) -> Solution:
        if constants is None:
            k = guess.k
        else:
            k = np.asarray(constants)

        t, x, lam, u, p, nu0, nuf = guess.t, guess.x, guess.lam, guess.u, guess.p, guess.nu0, guess.nuf

        t_norm, t0, tf = self._normalize_time(t)

        res_x, res_lam = self._compute_dynamic_residual(t, x, lam, u, p, k)

        return res_x, res_lam


    @staticmethod
    def _normalize_time(t: np.ndarray) -> Tuple[np.ndarray, float, float]:
        t0, tf = t[0], t[-1]
        return (t - t0) / (tf - t0), t0, tf

    @staticmethod
    def _restore_time(tau: np.ndarray, t0, tf) -> np.ndarray:
        return (tf - t0) * tau + t0

    def _compute_dynamic_residual(self, t_nodes, x_nodes, lam_nodes, u_nodes, p, k):
        # Simpson Method (3-Stage Lobatto IIIa)

        h_array = np.diff(t_nodes)

        t_mid = (t_nodes[:-1] + t_nodes[1:]) / 2
        u_mid = (u_nodes[:, :-1] + u_nodes[:, 1:]) / 2

        x_dot_nodes = self._compute_x_dot(t_nodes, x_nodes, u_nodes, p, k)

        x_mid = (x_nodes[:, :-1] + x_nodes[:, 1:]) / 2 - h_array / 8 * np.diff(x_dot_nodes)
        x_dot_mid = self._compute_x_dot(t_mid, x_mid, u_mid, p, k)

        res_x = np.diff(x_nodes) - h_array \
            * (1 / 6 * (x_dot_nodes[:, :-1] + x_dot_nodes[:, 1:]) + 2 / 3 * x_dot_mid)

        lam_dot_nodes = self._compute_lam_dot(t_nodes, x_nodes, lam_nodes, u_nodes, p, k)

        lam_mid = 1 / 2 * (lam_nodes[:, :-1] + lam_nodes[:, 1:]) - 1 / 8 * h_array * np.diff(lam_dot_nodes)
        lam_dot_mid = self._compute_lam_dot(t_mid, x_mid, lam_mid, u_mid, p, k)

        res_lam = np.diff(lam_nodes) - h_array \
            * (1 / 6 * (lam_dot_nodes[:, :-1] + lam_dot_nodes[:, 1:]) + 2 / 3 * lam_dot_mid)

        return res_x.flatten(), res_lam.flatten()

    def _compute_sensitivity_matrix(self):
        ...

    def _compute_step(self):
        ...
