from typing import Union, Callable, TypeVar

import casadi as ca
import numpy as np
from numpy import ndarray
from scipy.integrate import solve_bvp

from giuseppe.io import Solution
from ...problems.bvp import AdiffBVP
from ...problems.dual import AdiffDualOCP
from ...problems.dual.adiff import AdiffDiffControlHandler
from ...problems.components.adiff import maybe_expand
from ...utils.mixins import Picky
from ...utils.typing import NPArray

_scipy_bvp_sol = TypeVar('_scipy_bvp_sol')
_dyn_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_bc_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_preprocess_type = Callable[[Solution], tuple[NPArray, NPArray, NPArray]]
_postprocess_type = Callable[[_scipy_bvp_sol, NPArray], Solution]


class AdiffScipySolveBVP(Picky):
    """
    Class to use SciPy's BVP solver from scipy.integrate.solve_bvp

    The class takes in a supported problem type and wraps the boundary condition and dynamics functions to work.
    The class will also generate pre- and post-processing methods so that it can take in and output solutions in
    Giuseppe's native formats.

    """

    SUPPORTED_INPUTS = Union[AdiffBVP, AdiffDualOCP]

    def __init__(self, bvp: SUPPORTED_INPUTS,
                 tol: float = 0.001, bc_tol: float = 0.001, max_nodes: int = 1000, verbose: bool = False,
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
        """

        Picky.__init__(self, bvp)

        self.tol: float = tol
        self.bc_tol: float = bc_tol
        self.max_nodes: int = max_nodes
        self.verbose: bool = verbose
        self.use_jac: bool = use_jac

        dynamics, dyn_jac, boundary_conditions, bc_jac, preprocess, postprocess = self.load_problem(bvp)
        self.dynamics: _dyn_type = dynamics
        self.dyn_jac = dyn_jac
        self.boundary_conditions: _bc_type = boundary_conditions
        self.bc_jac = bc_jac
        self.preprocess: _preprocess_type = preprocess
        self.postprocess: _postprocess_type = postprocess

    def load_problem(self, bvp: SUPPORTED_INPUTS) -> tuple[Callable, Callable,
                                                           _bc_type, Callable, _preprocess_type, _postprocess_type]:
        if type(bvp) is AdiffBVP:
            dynamics = self._generate_bvp_dynamics(bvp)
            dyn_jac = None
            boundary_conditions = self._generate_bvp_bcs(bvp)
            bc_jac = None
            preprocess = self._preprocess_bvp_sol
            postprocess = self._postprocess_bvp_sol
        elif type(bvp) is AdiffDualOCP:
            if type(bvp.control_handler) is AdiffDiffControlHandler:
                dynamics = self._generate_ocp_diff_dynamics(bvp)
                dyn_jac = self._generate_ocp_diff_dynamics_jac(bvp)
                boundary_conditions = self._generate_ocp_diff_bcs(bvp)
                bc_jac = self._generate_ocp_diff_bc_jac(bvp)
                preprocess = self._preprocess_ocp_diff_sol
                postprocess = self._generate_postprocess_ocp_diff_sol(bvp)
            else:
                raise TypeError('To use Scipy\'s BVP Solver, problem needs a differential control handler')
        else:
            raise TypeError

        return dynamics, dyn_jac, boundary_conditions, bc_jac, preprocess, postprocess

    @staticmethod
    def _generate_bvp_dynamics(bvp: AdiffBVP) -> _dyn_type:
        bvp_dyn = maybe_expand(bvp.ca_dynamics)
        n_p = bvp.num_parameters

        def dynamics(tau_vec: NPArray, x_vec: NPArray, p: NPArray, k: NPArray) -> NPArray:
            t0, tf = p[-2], p[-1]
            tau_mult = (tf - t0)
            t_vec = tau_vec * tau_mult + t0
            map_size = len(tau_vec)

            p = p[:n_p]
            # p_vec = np.linspace(p, p, map_size).T
            # k_vec = np.linspace(k, k, map_size).T

            bvp_dyn_map = bvp_dyn.map(map_size)
            x_dot = np.asarray(bvp_dyn_map(t_vec, x_vec, p, k))
            return x_dot * tau_mult

        return dynamics

    @staticmethod
    def _generate_bvp_bcs(bvp: AdiffBVP) -> _bc_type:
        bvp_bc0 = maybe_expand(bvp.ca_boundary_conditions.initial)
        bvp_bcf = maybe_expand(bvp.ca_boundary_conditions.terminal)

        n_p = bvp.num_parameters

        def boundary_conditions(x0: NPArray, xf: NPArray, p: NPArray, k: NPArray):
            t0, tf = p[-2], p[-1]
            p = p[:n_p]
            return np.concatenate((np.asarray(bvp_bc0(t0, x0, p, k)).flatten(),
                                   np.asarray(bvp_bcf(tf, xf, p, k)).flatten()))

        return boundary_conditions

    @staticmethod
    def _preprocess_bvp_sol(sol: Solution) -> tuple[NPArray, NPArray, NPArray]:
        t0, tf = sol.t[0], sol.t[-1]
        p_guess = np.concatenate((sol.p, np.array([t0, tf])))
        tau_guess = (sol.t - t0) / (tf - t0)
        return tau_guess, sol.x, p_guess

    @staticmethod
    def _postprocess_bvp_sol(scipy_sol: _scipy_bvp_sol, k: NPArray) -> Solution:
        tau: NPArray = scipy_sol.x
        x: NPArray = scipy_sol.y
        p: NPArray = scipy_sol.p

        t0, tf = p[-2], p[-1]
        t = (tf - t0) * tau + t0
        p = p[:-2]

        return Solution(t=t, x=x, p=p, k=k, converged=scipy_sol.success)

    @staticmethod
    def _generate_ocp_diff_dynamics(dual_ocp: AdiffDualOCP) -> _dyn_type:
        args = dual_ocp.dual.args['dynamic']
        arg_names = dual_ocp.dual.arg_names['dynamic']
        _x_dot = dual_ocp.ocp.ca_dynamics(*dual_ocp.ocp.args)
        _lam_dot = dual_ocp.dual.ca_costate_dynamics(*args)
        _u_dot = dual_ocp.control_handler.ca_control_dynamics(*args)
        y_dyn = ca.Function('dy_dt', args, (ca.vcat((_x_dot, _lam_dot, _u_dot)),), arg_names, ('dy_dt',))

        y_dyn = maybe_expand(y_dyn)

        n_x = dual_ocp.dual.num_states
        n_lam = dual_ocp.dual.num_costates
        n_p = dual_ocp.dual.num_parameters

        ind_nu00 = n_p
        ind_lam0 = n_x
        ind_u0 = n_x + n_lam

        def dynamics(tau_vec: NPArray, y_vec: NPArray, p: NPArray, k: NPArray) -> NPArray:
            t0, tf = p[-2], p[-1]
            tau_mult = (tf - t0)
            t_vec = tau_vec * tau_mult + t0
            p = p[:ind_nu00]

            map_size = len(tau_vec)
            # p_vec = np.linspace(p, p, map_size).T
            # k_vec = np.linspace(k, k, map_size).T

            x_vec = y_vec[:ind_lam0, :]
            lam_vec = y_vec[ind_lam0:ind_u0, :]
            u_vec = y_vec[ind_u0:, :]

            y_dyn_map = y_dyn.map(map_size)

            y_dot = np.asarray(y_dyn_map(t_vec, x_vec, lam_vec, u_vec, p, k))

            return y_dot * tau_mult

        return dynamics

    @staticmethod
    def _generate_ocp_diff_dynamics_jac(dual_ocp: AdiffDualOCP):
        df_dy = maybe_expand(dual_ocp.df_dy)
        df_dp = maybe_expand(dual_ocp.df_dp)

        def dynamics_jac(tau_vec: NPArray, y_vec: NPArray, p_nu_t: NPArray, k: NPArray):

            map_size = len(tau_vec)

            df_dy_map = df_dy.map(map_size)
            df_dp_map = df_dp.map(map_size)

            df_dy_vectorized = np.asarray(tuple(df_dy_map(tau_vec, y_vec, p_nu_t, k))).transpose((1, 0, 2))
            df_dp_vectorized = np.asarray(tuple(df_dp_map(tau_vec, y_vec, p_nu_t, k))).transpose((1, 0, 2))

            return df_dy_vectorized, df_dp_vectorized

        return dynamics_jac

    @staticmethod
    def _generate_ocp_diff_bcs(dual_ocp: AdiffDualOCP) -> _bc_type:
        ocp_bc0 = maybe_expand(dual_ocp.ocp.ca_boundary_conditions.initial)
        ocp_bcf = maybe_expand(dual_ocp.ocp.ca_boundary_conditions.terminal)

        dual_bc0 = maybe_expand(dual_ocp.dual.ca_adj_boundary_conditions.initial)
        dual_bcf = maybe_expand(dual_ocp.dual.ca_adj_boundary_conditions.terminal)

        control_bc = maybe_expand(dual_ocp.control_handler.ca_control_bc)

        n_x = dual_ocp.dual.num_states
        n_lam = dual_ocp.dual.num_costates
        n_p = dual_ocp.dual.num_parameters
        n_nu0 = dual_ocp.dual.num_initial_adjoints

        ind_lam0 = n_x
        ind_u0 = n_x + n_lam
        ind_nu00 = n_p
        ind_nuf0 = n_p + n_nu0

        def boundary_conditions(y0: NPArray, yf: NPArray, _p: NPArray, k: NPArray):
            x0 = y0[:ind_lam0]
            lam0 = y0[ind_lam0:ind_u0]
            u0 = y0[ind_u0:]

            xf = yf[:ind_lam0]
            lamf = yf[ind_lam0:ind_u0]
            uf = yf[ind_u0:]

            t0, tf = _p[-2], _p[-1]
            p = _p[:ind_nu00]
            nu0 = _p[ind_nu00:ind_nuf0]
            nuf = _p[ind_nuf0:-2]

            residual = np.concatenate((
                np.asarray(ocp_bc0(t0, x0, u0, p, k)).flatten(),
                np.asarray(ocp_bcf(tf, xf, uf, p, k)).flatten(),
                np.asarray(dual_bc0(t0, x0, lam0, u0, p, nu0, k)).flatten(),
                np.asarray(dual_bcf(tf, xf, lamf, uf, p, nuf, k)).flatten(),
                np.asarray(control_bc(t0, x0, lam0, u0, p, k)).flatten()
            ))

            return residual

        return boundary_conditions

    @staticmethod
    def _generate_ocp_diff_bc_jac(dual_ocp: AdiffDualOCP):
        dbc_dya = maybe_expand(dual_ocp.dbc_dya)
        dbc_dyb = maybe_expand(dual_ocp.dbc_dyb)
        dbc_dp = maybe_expand(dual_ocp.dbc_dp)

        def bc_jac(y0: NPArray, yf: NPArray, p_nu_t: NPArray, k: NPArray):
            return np.asarray(dbc_dya(y0, yf, p_nu_t, k)), \
                   np.asarray(dbc_dyb(y0, yf, p_nu_t, k)), \
                   np.asarray(dbc_dp(y0, yf, p_nu_t, k))

        return bc_jac

    @staticmethod
    def _preprocess_ocp_diff_sol(sol: Solution) -> tuple[NPArray, NPArray, NPArray]:
        t0, tf = sol.t[0], sol.t[-1]
        tau_guess = (sol.t - t0) / (tf - t0)
        p_guess = np.concatenate((sol.p, sol.nu0, sol.nuf, np.array([t0, tf])))
        y = np.vstack((sol.x, sol.lam, sol.u))
        return tau_guess, y, p_guess

    @staticmethod
    def _generate_postprocess_ocp_diff_sol(dual_ocp: AdiffDualOCP) \
            -> Callable[[_scipy_bvp_sol, ndarray], Solution]:

        n_x = dual_ocp.ocp.num_states
        n_p = dual_ocp.ocp.num_parameters
        n_u = dual_ocp.ocp.num_controls

        n_lam = dual_ocp.dual.num_costates
        n_nu_0 = dual_ocp.dual.num_initial_adjoints
        n_nu_f = dual_ocp.dual.num_terminal_adjoints

        def _postprocess_ocp_diff_sol(scipy_sol: _scipy_bvp_sol, k: NPArray) -> Solution:
            tau: NPArray = scipy_sol.x
            t0, tf = scipy_sol.p[-2], scipy_sol.p[-1]
            t: NPArray = (tf - t0) * tau + t0

            x: NPArray = scipy_sol.y[:n_x]
            lam: NPArray = scipy_sol.y[n_x:n_x + n_lam]
            u: NPArray = scipy_sol.y[n_x + n_lam:n_x + n_lam + n_u]

            p: NPArray = scipy_sol.p[:n_p]
            nu_0: NPArray = scipy_sol.p[n_p:n_p + n_nu_0]
            nu_f: NPArray = scipy_sol.p[n_p + n_nu_0:n_p + n_nu_0 + n_nu_f]

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu_0, nuf=nu_f, k=k, converged=scipy_sol.success)

        return _postprocess_ocp_diff_sol

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

        sol: _scipy_bvp_sol = solve_bvp(
                lambda tau_vec, x_vec, p: self.dynamics(tau_vec, x_vec, p, k),
                lambda x0, xf, p: self.boundary_conditions(x0, xf, p, k),
                tau_guess, x_guess, p_guess,
                fun_jac=_fun_jac,
                bc_jac=_bc_jac,
                tol=self.tol, bc_tol=self.bc_tol, max_nodes=self.max_nodes, verbose=self.verbose
        )

        return self.postprocess(sol, k)
