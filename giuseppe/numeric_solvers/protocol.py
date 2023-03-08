from typing import Union, Protocol, runtime_checkable, Optional

import numpy as np

from giuseppe.data_classes import Solution


@runtime_checkable
class NumericSolver(Protocol):
    """
    Protocol for Numeric Solvers in Giuseppe

    The class takes in a supported problem type and wraps the boundary condition and dynamics functions to work.
    The class will also generate pre- and post-processing methods so that it can take in and output solutions in
    Giuseppe's native formats.

    """

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
        ...
