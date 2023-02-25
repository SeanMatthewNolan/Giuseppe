from typing import Union, Protocol, runtime_checkable

import numpy as np
from scipy.integrate import solve_bvp

from giuseppe.io import Solution
from giuseppe.problems.protocols import Problem
from giuseppe.problems.conversions import convert_dual_to_bvp


@runtime_checkable
class NumericSolver(Protocol):
    """
    Protocol for Numeric Solvers in Giuseppe

    The class takes in a supported problem type and wraps the boundary condition and dynamics functions to work.
    The class will also generate pre- and post-processing methods so that it can take in and output solutions in
    Giuseppe's native formats.

    """

    def solve(self, constants: np.ndarray, guess: Solution) -> Solution:
        """
        Solve pre-loaded problem

        Parameters
        ----------
        constants : np.ndarray
            array of constants which define the problem numerically
        guess : Solution or DualOCPSol
            previous solution (or approximate solution) to serve as guess for BVP solver

        Returns
        -------
        solution : Solution
            solution to the problem for given constants

        """
        ...
