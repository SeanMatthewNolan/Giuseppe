from __future__ import annotations

import sys
from typing import Callable, TypeVar, Protocol

import numpy as np
from scipy.integrate import solve_bvp

from giuseppe.io.solution import Solution
from giuseppe.problems.bvp import CompBVP
from giuseppe.problems.dual import CompDualOCP
from giuseppe.utils.typing import NPArray

_scipy_bvp_sol = TypeVar('_scipy_bvp_sol')
if sys.version_info >= (3, 10):
    _dyn_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
    _bc_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
    _preprocess_type = Callable[[Solution], tuple[NPArray, NPArray, NPArray]]
    _postprocess_type = Callable[[_scipy_bvp_sol, NPArray], Solution]
else:
    _dyn_type = Callable
    _bc_type = Callable
    _preprocess_type = Callable
    _postprocess_type = Callable


class SciPyBVP(Protocol):
    def dynamics(self, tau_vec: NPArray, x_vec: NPArray, p: NPArray, k: NPArray) -> NPArray:
        ...

    def boundary_conditions(self, x0: NPArray, xf: NPArray, p: NPArray, k: NPArray):
        ...

    def preprocess(self, guess: Solution) -> tuple[NPArray, NPArray, NPArray]:
        ...

    def postprocess(self, scipy_sol: _scipy_bvp_sol) -> Solution:
        ...


class SciPySolver:
    """
    Class to use SciPy's BVP solver from scipy.integrate.solve_bvp

    The class takes in a supported problem type and wraps the boundary condition and dynamics functions to work.
    The class will also generate pre- and post-processing methods so that it can take in and output solutions in
    Giuseppe's native formats.

    """

    def __init__(self, prob: SciPyBVP,
                 tol: float = 0.001, bc_tol: float = 0.001, max_nodes: int = 1000, verbose: bool = False):
        """
        Initialize ScipySolveBVP

        Parameters
        ----------
        prob : CompBVP or CompDualOCP
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
        self.verbose: bool = verbose

        self.prob: SciPyBVP = prob

    def solve(self, constants: NPArray, guess: Solution) -> Solution:
        """
        Solve BVP (or dualized OCP) with instance of ScipySolveBVP

        Parameters
        ----------
        constants : np.ndarray
            array of constants which define the problem numerically
        guess : Solution or DualOCPSol
            previous solution (or approximate solution) to serve as guess for BVP solver

        Returns
        -------
        solution : Solution or DualOCPSol
            solution to the BVP for given constants

        """

        tau_guess, x_guess, p_guess = self.prob.preprocess(guess)
        k = constants

        sol: _scipy_bvp_sol = solve_bvp(
                lambda tau, x, p: self.prob.dynamics(tau, x, p, k),
                lambda x0, xf, p: self.prob.boundary_conditions(x0, xf, p, k),
                tau_guess, x_guess, p_guess,
                tol=self.tol, bc_tol=self.bc_tol, max_nodes=self.max_nodes, verbose=self.verbose
        )

        return self.prob.postprocess(sol)
