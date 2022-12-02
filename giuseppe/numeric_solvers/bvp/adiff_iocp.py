from typing import Union, Callable, TypeVar

import casadi as ca
import numpy as np
from numpy import ndarray
from .python_solver import solve_bvp

from giuseppe.io import Solution
from ...problems.bvp import AdiffBVP
from ...problems.dual import AdiffDualOCP
from ...problems.dual.adiff import AdiffDiffControlHandler
from ...problems.components.adiff import maybe_expand
from ...utils.mixins import Picky
from ...utils.typing import NPArray
from .adiff_scipy import AdiffScipySolveBVP

_scipy_bvp_sol = TypeVar('_scipy_bvp_sol')
_dyn_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_bc_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_preprocess_type = Callable[[Solution], tuple[NPArray, NPArray, NPArray]]
_postprocess_type = Callable[[_scipy_bvp_sol, NPArray], Solution]


class AdiffPythonSolveBVP(AdiffScipySolveBVP):
    """
    Class to use SciPy's BVP solver from scipy.integrate.solve_bvp

    The class takes in a supported problem type and wraps the boundary condition and dynamics functions to work.
    The class will also generate pre- and post-processing methods so that it can take in and output solutions in
    Giuseppe's native formats.

    """

    SUPPORTED_INPUTS = Union[AdiffBVP, AdiffDualOCP]

    def __init__(self, bvp: SUPPORTED_INPUTS,
                 tol: float = 0.001, bc_tol: float = 0.001, max_nodes: int = 1000, verbose: Union[bool, int] = False,
                 use_jac: bool = False):
        """
        Initialize ScipySolveBVP

        Parameters
        ----------
        bvp : CompBVP or CompDualOCP
            the BVP (or dualized OCP) to solve
        tol : float, default=0.001
            sets `tol` kwarg for `scipy.integrate.solve_bvp`
        bc_tol : float, default=0.001
            sets `bc_tol` kwarg for `scipy.integrate.solve_bvp`
        max_nodes: int, default=1000
            sets `max_nodes` kwarg for `scipy.integrate.solve_bvp`
        verbose : bool, default=False
            sets `verbose` kwarg for `scipy.integrate.solve_bvp`
        use_jac : bool, default=False
            Whether to use CasADi-generated AD jac functions in BVP solver
        """
        super().__init__(bvp, tol, bc_tol, max_nodes, verbose, use_jac)
        self.bounding_function = self.load_additional_features(bvp)

    def load_additional_features(self, bvp: SUPPORTED_INPUTS) -> Callable:
        if type(bvp) is AdiffBVP:
            # TODO: Implement
            bounding_fun = None
        elif type(bvp) is AdiffDualOCP:
            bounding_fun = self._generate_bounding_function(bvp)
        else:
            raise TypeError

        return bounding_fun

    @staticmethod
    def _generate_bounding_function(bvp: AdiffDualOCP):
        def _bounding_fun(tau_vec, y_vec, p, k):
            # TODO p should be probably be bounded based on y_a, y_b, p, k...
            #  but this requires major reworking of implementation to allow user to explicitly define function of
            #  initial/final state. For now, bounded based on y_a only.
            map_size = len(tau_vec)
            _y_bnd_map = bvp.y_bnd_func.map(map_size)
            # _p_bnd_map = bvp.p_bnd_func.map(map_size)

            _y_bnd = np.asarray(_y_bnd_map(tau_vec, y_vec, p, k))
            # _p_bnd = np.asarray(_p_bnd_map(tau_vec, y_vec, p, k))
            _p_bnd = np.asarray(bvp.p_bnd_func(tau_vec[0], y_vec[:, 0], p, k)).flatten()
            return _y_bnd, _p_bnd
        return _bounding_fun

    def solve(self, constants: NPArray, guess: Solution) -> Solution:
        """
        Solve BVP (or dualized OCP) with instance of ScipySolveBVP

        Parameters
        ----------
        constants : np.ndarray
            array of constants which define the problem numerically
        guess : BVPSol or DualOCPSol
            previous solution (or approximate solution) to serve as guess for BVP solver

        Returns
        -------
        solution : BVPSol or DualOCPSol
            solution to the BVP for given constants

        """

        tau_guess, x_guess, p_guess = self.preprocess(guess)
        k = constants

        if self.use_jac:
            def _fun_jac(tau_vec, x_vec, p):
                return self.dyn_jac(tau_vec, x_vec, p, k)

            def _bc_jac(x0, xf, p):
                return self.bc_jac(x0, xf, p, k)
        else:
            _fun_jac = None
            _bc_jac = None

        if self.bounding_function is not None:
            def _bounding_fun(tau_vec, x_vec, p):
                return self.bounding_function(tau_vec, x_vec, p, k)
        else:
            _bounding_fun = None

        sol: _scipy_bvp_sol = solve_bvp(
            lambda tau_vec, x_vec, p: self.dynamics(tau_vec, x_vec, p, k),
            lambda x0, xf, p: self.boundary_conditions(x0, xf, p, k),
            tau_guess, x_guess, p_guess,
            fun_jac=_fun_jac, bc_jac=_bc_jac, bounding_fun=_bounding_fun,
            tol=self.tol, bc_tol=self.bc_tol, max_nodes=self.max_nodes, verbose=self.verbose
        )

        return self.postprocess(sol, k)