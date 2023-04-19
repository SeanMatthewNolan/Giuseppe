from typing import Union, Optional

import numpy as np
from scipy.integrate import solve_bvp

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import BVP, Dual
from giuseppe.problems.conversions import convert_dual_to_bvp, vectorize
from .scipy_bvp_problem import SciPyBVP
from .scipy_types import _scipy_bvp_sol


class SciPySolver:
    """
    Class to use SciPy's BVP solver from scipy.integrate.solve_bvp

    The class takes in a supported problem type and wraps the boundary condition and dynamics functions to work.
    The class will also generate pre- and post-processing methods so that it can take in and output solutions in
    Giuseppe's native formats.

    """

    def __init__(self, prob: Union[BVP, Dual], use_jit_compile: bool = True, perform_vectorize: bool = True,
                 tol: float = 0.001, bc_tol: float = 0.001, max_nodes: int = 1000, node_buffer: int = 20,
                 verbose: Union[bool, int] = False):
        """
        Initialize SciPySolver

        Parameters
        ----------
        prob : BVP, Dual
            the BVP (or dualized OCP) to solve
        tol : float, default=0.001
            sets `tol` kwarg for `scipy.integrate.solve_bvp`
        bc_tol : float, default=0.001
            sets `bc_tol` kwarg for `scipy.integrate.solve_bvp`
        max_nodes: int, default=1000
            sets `max_nodes` kwarg for `scipy.integrate.solve_bvp`
        verbose : bool, default=False
            sets `verbose` kwarg for `scipy.integrate.solve_bvp`
        """

        self.tol: float = tol
        self.bc_tol: float = bc_tol
        self.max_nodes: int = max_nodes
        self.node_buffer: int = node_buffer
        self.verbose: Union[bool, int] = verbose

        if prob.prob_class == 'dual':
            prob = convert_dual_to_bvp(prob, perform_vectorize=perform_vectorize)

        self.prob: SciPyBVP = SciPyBVP(prob, use_jit_compile=use_jit_compile)

    def solve(self, guess: Solution, constants: Optional[np.ndarray] = None) -> Solution:
        """
        Solve BVP (or dualized OCP) with instance of ScipySolveBVP

        Parameters
        ----------
        guess : Solution
            previous solution (or approximate solution) to serve as guess for BVP solver
        constants : np.ndarray, optional
            array of constants which define the problem numerically, if not given solver will use constants from guess

        Returns
        -------
        solution : Solution
            solution to the BVP for given constants

        """

        tau_guess, x_guess, p_guess = self.prob.preprocess(guess)

        if constants is None:
            constants = guess.k

        constants = np.asarray(constants)
        max_nodes = max(self.max_nodes, len(tau_guess) + self.node_buffer)

        try:
            sol: _scipy_bvp_sol = solve_bvp(
                    lambda tau, x, p: self.prob.compute_dynamics(tau, x, p, constants),
                    lambda x0, xf, p: self.prob.compute_boundary_conditions(x0, xf, p, constants),
                    tau_guess, x_guess, p_guess,
                    tol=self.tol, bc_tol=self.bc_tol, max_nodes=max_nodes, verbose=self.verbose
            )

        except (ZeroDivisionError, np.linalg.LinAlgError):
            sol: _scipy_bvp_sol = self._form_solution_when_solver_throws_exception(tau_guess, x_guess, p_guess)

        return self.prob.post_process(sol, constants)

    @staticmethod
    def _form_solution_when_solver_throws_exception(_x, _y, _p) -> _scipy_bvp_sol:
        class _FailedSciPySol:
            x: np.ndarray
            y: np.ndarray
            p: np.ndarray
            success: bool = False

        _failed = _FailedSciPySol
        _failed.x, _failed.y, _failed.p = _x, _y, _p
        _failed.status = 3

        return _failed
