"""
This solver follows the algorithms given in

"Parallel collocation solution of index-1 BVP-DAEs arising from constrained optimal control problems" - Fabien

and

"A BVP Solver Based on Residual Control and the MATLAB PSE

"""

from typing import Optional, Tuple
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from giuseppe.data_classes import Solution, remesh
from giuseppe.problems.protocols import Dual, VectorizedDual
from giuseppe.problems.conversions import vectorize
from giuseppe.utils.quadratures import SimpsonQuadrature

from ...protocol import NumericSolver


@dataclass
class ProblemData:
    t_nodes: np.ndarray
    t_canon: np.ndarray
    h: np.ndarray
    x_nodes: np.ndarray
    x_canon: np.ndarray
    x_dot_canon: np.ndarray
    lam_nodes: np.ndarray
    lam_canon: np.ndarray
    lam_dot_canon: np.ndarray
    u_canon: np.ndarray
    p: np.ndarray
    nu0: np.ndarray
    nuf: np.ndarray
    k: np.ndarray

    def to_q(self) -> np.ndarray:
        q = np.concatenate((
            self.x_nodes[:, :-1].flatten(),
            self.lam_nodes[:, :-1].flatten(),
            self.x_dot_canon.flatten(),
            self.lam_dot_canon.flatten(),
            self.u_canon.flatten(),
            self.x_nodes[:, -1].flatten(),
            self.lam_nodes[:, -1].flatten(),
            self.p,
            self.nu0,
            self.nuf,
            (self.t_nodes[0], self.t_nodes[-1])
        ))

        return q


class DualCollocation(NumericSolver):
    def __init__(self, prob: Dual):
        if not isinstance(prob, VectorizedDual):
            prob: VectorizedDual = vectorize(prob)

        self.max_steps = 10

        self.quadrature = SimpsonQuadrature()

        self.prob: VectorizedDual = deepcopy(prob)

        self._n_x = self.prob.num_states
        self._n_lam = self.prob.num_costates
        self._n_u = self.prob.num_controls

        self._n_p = self.prob.num_parameters
        self._n_nu0 = self.prob.num_initial_adjoints
        self._n_nuf = self.prob.num_terminal_adjoints

        self._compute_x_dot = self.prob.compute_dynamics_vectorized
        self._compute_lam_dot = self.prob.compute_costate_dynamics_vectorized
        self._compute_dh_du = self.prob.compute_control_law_vectorized

        self._compute_psi_0 = self.prob.compute_initial_boundary_conditions
        self._compute_psi_f = self.prob.compute_terminal_boundary_conditions

        self._compute_adj_bc_0 = self.prob.compute_initial_adjoint_boundary_conditions
        self._compute_adj_bc_f = self.prob.compute_terminal_adjoint_boundary_conditions

    def solve(self, guess: Solution, constants: Optional[np.ndarray] = None) -> Solution:
        # Initialize Problem
        data = self._initialize_from_guess(guess)

        if constants is None:
            data.k = guess.k
        else:
            data.k = np.asarray(constants)

        # Perform Iterations
        for step_num in range(self.max_steps):
            ...

        return self._compute_f(data)

    def _initialize_from_guess(self, guess) -> ProblemData:
        t_nodes, x_nodes, lam_nodes, u_nodes, p, nu0, nuf, k \
            = guess.t, guess.x, guess.lam, guess.u, guess.p, guess.nu0, guess.nuf, guess.k

        t0, tf = t_nodes[0], t_nodes[-1]
        # t_norm = (t - t0) / (tf - t0), t0, tf

        h = np.diff(t_nodes)

        # Canonical Points
        t_canon = np.concatenate([
            t_nodes[:-1] + h * rho_i for rho_i in self.quadrature.rho
        ])

        guess_canon = remesh(guess, t_canon)
        x_canon, lam_canon, u_canon = guess_canon.x, guess_canon.lam, guess_canon.u

        x_dot_canon = self._compute_x_dot(t_canon, x_canon, u_canon, p, guess.k)
        lam_dot_canon = self._compute_lam_dot(t_canon, x_canon, lam_canon, u_canon, p, guess.k)

        data = ProblemData(
                t_nodes, t_canon, h, x_nodes, x_canon, x_dot_canon, lam_nodes, lam_canon, lam_dot_canon, u_canon,
                p, nu0, nuf, k
        )

        return data

    def _compute_f(self, data: ProblemData) -> np.ndarray:
        return np.hstack((
            (self._compute_x_dot(data.t_canon, data.x_canon, data.u_canon, data.p, data.k)
             - data.x_dot_canon).flatten(),
            (self._compute_lam_dot(data.t_canon, data.x_canon, data.lam_canon, data.u_canon, data.p, data.k)
             - data.lam_dot_canon).flatten(),
            self._compute_dh_du(data.t_canon, data.x_canon, data.lam_canon, data.u_canon, data.p, data.k).flatten(),
            self._compute_quadrature(data.h, data.x_nodes, data.x_dot_canon).flatten(),
            self._compute_quadrature(data.h, data.lam_nodes, data.lam_dot_canon).flatten(),
            self._compute_psi_0(data.t_nodes[0], data.x_nodes[:, 0], data.p, data.k).flatten(),
            self._compute_psi_f(data.t_nodes[-1], data.x_nodes[:, -1], data.p, data.k).flatten(),
            self._compute_adj_bc_0(
                    data.t_nodes[0], data.x_nodes[:, 0], data.lam_nodes[:, 0], data.u_canon[:, 0],
                    data.p, data.nu0, data.k).flatten(),
            self._compute_adj_bc_f(
                    data.t_nodes[-1], data.x_nodes[:, -1], data.lam_nodes[:, -1], data.u_canon[:, -1],
                    data.p, data.nuf, data.k).flatten(),
        ))

    def _compute_quadrature(self, h, _y_nodes, _y_dot_canon) -> np.ndarray:
        return np.sum(_y_dot_canon.reshape((3, 3, 10)).swapaxes(1, 2) * self.quadrature.beta, axis=2) * h \
            - np.diff(_y_nodes)

    @staticmethod
    def _normalize_time(t: np.ndarray) -> Tuple[np.ndarray, float, float]:
        t0, tf = t[0], t[-1]
        return (t - t0) / (tf - t0), t0, tf

    @staticmethod
    def _restore_time(tau: np.ndarray, t0, tf) -> np.ndarray:
        return (tf - t0) * tau + t0

