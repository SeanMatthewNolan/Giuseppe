from typing import Union, Callable, TypeVar, Optional
from warnings import warn

import numpy as np
from numpy import ndarray
from scipy.integrate import solve_bvp

from giuseppe.io.solution import Solution
from ...problems.bvp import CompBVP
from ...problems.dual import CompDualOCP
from ...problems.dual.compiled import CompAlgControlHandler, CompDiffControlHandler, CompDiffControlHandlerNumeric
from ...utils.complilation import jit_compile
from ...utils.mixins import Picky
from ...utils.typing import NumbaArray, NumbaMatrix, NPArray

_scipy_bvp_sol = TypeVar('_scipy_bvp_sol')
_dyn_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_bc_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_preprocess_type = Callable[[Solution], tuple[NPArray, NPArray, NPArray]]
_postprocess_type = Callable[[_scipy_bvp_sol, NPArray], Solution]


class ScipySolveBVP(Picky):
    """
    Class to use SciPy's BVP solver from scipy.integrate.solve_bvp

    The class takes in a supported problem type and wraps the boundary condition and dynamics functions to work.
    The class will also generate pre- and post-processing methods so that it can take in and output solutions in
    Giuseppe's native formats.

    """

    SUPPORTED_INPUTS = Union[CompBVP, CompDualOCP]

    def __init__(self, bvp: SUPPORTED_INPUTS, use_jit_compile: Optional[bool] = None,
                 tol: float = 0.001, bc_tol: float = 0.001, max_nodes: int = 1000, verbose: bool = False):
        """
        Initialize ScipySolveBVP

        Parameters
        ----------
        bvp : CompBVP or CompDualOCP
            the BVP (or dualized OCP) to solve
        use_jit_compile : bool, optional, default=None
            whether to JIT compile wrapper function with Numba
            if None, the solver will match the given BVP
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

        # Options directly for scipy.solve_bvp
        if use_jit_compile is None:
            self.use_jit_compile: bool = bvp.use_jit_compile
        else:
            if not bvp.use_jit_compile and use_jit_compile:
                warn('Cannot JIT compile BVP solver with non-JIT compiled BVP! Setting use_jit_compile to False')
                self.use_jit_compile: bool = False
            else:
                self.use_jit_compile: bool = use_jit_compile

        self.tol: float = tol
        self.bc_tol: float = bc_tol
        self.max_nodes: int = max_nodes
        self.verbose: bool = verbose

        dynamics, boundary_conditions, preprocess, postprocess = self.load_problem(bvp)
        self.dynamics: _dyn_type = dynamics
        self.boundary_conditions: _bc_type = boundary_conditions
        self.preprocess: _preprocess_type = preprocess
        self.postprocess: _postprocess_type = postprocess

    def load_problem(self, bvp: SUPPORTED_INPUTS) -> tuple[_dyn_type, _bc_type, _preprocess_type, _postprocess_type]:
        if type(bvp) is CompBVP:
            dynamics = self._generate_bvp_dynamics(bvp)
            boundary_conditions = self._generate_bvp_bcs(bvp)
            preprocess = self._preprocess_bvp_sol
            postprocess = self._postprocess_bvp_sol
        elif type(bvp) is CompDualOCP:
            if type(bvp.control_handler) is CompAlgControlHandler:
                dynamics = self._generate_ocp_alg_dynamics(bvp)
                boundary_conditions = self._generate_ocp_alg_bcs(bvp)
                preprocess = self._preprocess_ocp_alg_sol
                postprocess = self._generate_postprocess_ocp_alg_sol(bvp)
            elif type(bvp.control_handler) in [CompDiffControlHandler, CompDiffControlHandlerNumeric]:
                dynamics = self._generate_ocp_diff_dynamics(bvp)
                boundary_conditions = self._generate_ocp_diff_bcs(bvp)
                preprocess = self._preprocess_ocp_diff_sol
                postprocess = self._generate_postprocess_ocp_diff_sol(bvp)
            else:
                raise TypeError('To use Scipy\'s BVP Solver, problem needs a algebraic or differential control handler')
        else:
            raise TypeError

        return dynamics, boundary_conditions, preprocess, postprocess

    def _generate_bvp_dynamics(self, bvp: CompBVP) -> _dyn_type:
        bvp_dyn = bvp.dynamics

        def dynamics(tau_vec: NPArray, x_vec: NPArray, p: NPArray, k: NPArray) -> NPArray:
            t0, tf = p[-2], p[-1]
            tau_mult = (tf - t0)
            t_vec = tau_vec * tau_mult + t0

            p = p[:-2]

            x_dot = np.empty_like(x_vec)  # Need to pre-allocate for Numba
            for idx, (ti, xi) in enumerate(zip(t_vec, x_vec.T)):
                x_dot[:, idx] = bvp_dyn(ti, xi, p, k)

            return x_dot * tau_mult

        if self.use_jit_compile:
            dynamics = jit_compile(dynamics, (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray))

        return dynamics

    def _generate_bvp_bcs(self, bvp: CompBVP) -> _bc_type:
        bvp_bc0 = bvp.boundary_conditions.initial
        bvp_bcf = bvp.boundary_conditions.terminal

        def boundary_conditions(x0: NPArray, xf: NPArray, p: NPArray, k: NPArray):
            t0, tf = p[-2], p[-1]
            p = p[:-2]
            return np.concatenate((np.asarray(bvp_bc0(t0, x0, p, k)), np.asarray(bvp_bcf(tf, xf, p, k))))

        if self.use_jit_compile:
            boundary_conditions = jit_compile(boundary_conditions, (NumbaArray, NumbaArray, NumbaArray, NumbaArray))

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

    def _generate_ocp_alg_dynamics(self, dual_ocp: CompDualOCP) -> _dyn_type:
        state_dyn = dual_ocp.comp_ocp.dynamics
        costate_dyn = dual_ocp.comp_dual.costate_dynamics

        n_x = dual_ocp.comp_ocp.num_states
        n_p = dual_ocp.comp_ocp.num_parameters
        n_lam = dual_ocp.comp_dual.num_costates

        control = dual_ocp.control_handler.control

        def dynamics(tau_vec: NPArray, y_vec: NPArray, p: NPArray, k: NPArray) -> NPArray:
            t0, tf = p[-2], p[-1]
            tau_mult = (tf - t0)
            t_vec = tau_vec * tau_mult + t0
            p = p[:n_p]

            y_dot = np.empty_like(y_vec)  # Need to pre-allocate for Numba
            for idx, (ti, yi) in enumerate(zip(t_vec, y_vec.T)):
                xi = yi[:n_x]
                lami = yi[n_x:n_x + n_lam]

                ui = control(t0, xi, lami, p, k)
                y_dot[:n_x, idx] = state_dyn(ti, xi, ui, p, k)
                y_dot[n_x:n_x + n_lam, idx] = costate_dyn(ti, xi, lami, ui, p, k)

            return y_dot * tau_mult

        if self.use_jit_compile:
            dynamics = jit_compile(dynamics, (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray))

        return dynamics

    def _generate_ocp_alg_bcs(self, dual_ocp: CompDualOCP) -> _bc_type:
        ocp_bc0 = dual_ocp.comp_ocp.boundary_conditions.initial
        ocp_bcf = dual_ocp.comp_ocp.boundary_conditions.terminal

        dual_bc0 = dual_ocp.comp_dual.adjoined_boundary_conditions.initial
        dual_bcf = dual_ocp.comp_dual.adjoined_boundary_conditions.terminal

        n_x = dual_ocp.comp_ocp.num_states
        n_p = dual_ocp.comp_ocp.num_parameters
        n_lam = dual_ocp.comp_dual.num_costates
        n_nu_0 = dual_ocp.comp_dual.num_initial_adjoints
        n_nu_f = dual_ocp.comp_dual.num_terminal_adjoints

        control = dual_ocp.control_handler.control

        def boundary_conditions(y0: NPArray, yf: NPArray, _p: NPArray, k: NPArray):
            x0 = y0[:n_x]
            lam0 = y0[n_x:n_x + n_lam]

            xf = yf[:n_x]
            lamf = yf[n_x:n_x + n_lam]

            p = _p[:n_p]
            nu0 = _p[n_p:n_p + n_nu_0]
            nuf = _p[n_p + n_nu_0: n_p + n_nu_0 + n_nu_f]
            t0, tf = _p[-2], _p[-1]

            u0 = control(t0, x0, lam0, p, k)
            uf = control(tf, xf, lamf, p, k)

            return np.concatenate((
                np.asarray(ocp_bc0(t0, x0, u0, p, k)), np.asarray(ocp_bcf(tf, xf, uf, p, k)),
                np.asarray(dual_bc0(t0, x0, lam0, u0, p, nu0, k)), np.asarray(dual_bcf(tf, xf, lamf, uf, p, nuf, k))
            ))

        if self.use_jit_compile:
            boundary_conditions = jit_compile(boundary_conditions, (NumbaArray, NumbaArray, NumbaArray, NumbaArray))

        return boundary_conditions

    @staticmethod
    def _preprocess_ocp_alg_sol(sol: Solution) -> tuple[NPArray, NPArray, NPArray]:
        t0, tf = sol.t[0], sol.t[-1]
        tau_guess = (sol.t - t0) / (tf - t0)
        p_guess = np.concatenate((sol.p, sol.nu0, sol.nuf, np.array([t0, tf])))
        y = np.vstack((sol.x, sol.lam))
        return tau_guess, y, p_guess

    @staticmethod
    def _generate_postprocess_ocp_alg_sol(dual_ocp: CompDualOCP) -> Callable[[_scipy_bvp_sol, ndarray], Solution]:

        n_x = dual_ocp.comp_ocp.num_states
        n_p = dual_ocp.comp_ocp.num_parameters
        n_lam = dual_ocp.comp_dual.num_costates
        n_nu_0 = dual_ocp.comp_dual.num_initial_adjoints
        n_nu_f = dual_ocp.comp_dual.num_terminal_adjoints

        control = dual_ocp.control_handler.control

        def _postprocess_ocp_alg_sol(scipy_sol: _scipy_bvp_sol, k: NPArray) -> Solution:
            tau: NPArray = scipy_sol.x
            t0, tf = scipy_sol.p[-2], scipy_sol.p[-1]
            t: NPArray = (tf - t0) * tau + t0

            x: NPArray = scipy_sol.y[:n_x]
            lam: NPArray = scipy_sol.y[n_x:n_x + n_lam]

            p: NPArray = scipy_sol.p[:n_p]
            nu_0: NPArray = scipy_sol.p[n_p:n_p + n_nu_0]
            nu_f: NPArray = scipy_sol.p[n_p + n_nu_0:n_p + n_nu_0 + n_nu_f]

            u = np.array([control(ti, xi, lami, p, k) for ti, xi, lami in zip(t, x.T, lam.T)]).T

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu_0, nuf=nu_f, k=k, converged=scipy_sol.success)

        return _postprocess_ocp_alg_sol

    def _generate_ocp_diff_dynamics(self, dual_ocp: CompDualOCP) -> _dyn_type:
        state_dyn = dual_ocp.comp_ocp.dynamics
        costate_dyn = dual_ocp.comp_dual.costate_dynamics
        control_dyn = dual_ocp.control_handler.control_dynamics

        n_x = dual_ocp.comp_ocp.num_states
        n_p = dual_ocp.comp_ocp.num_parameters
        n_lam = dual_ocp.comp_dual.num_costates
        n_u = dual_ocp.comp_ocp.num_controls

        def dynamics(tau_vec: NPArray, y_vec: NPArray, p: NPArray, k: NPArray) -> NPArray:
            t0, tf = p[-2], p[-1]
            tau_mult = (tf - t0)
            t_vec = tau_vec * tau_mult + t0
            p = p[:n_p]

            y_dot = np.empty_like(y_vec)  # Need to pre-allocate for Numba
            for idx, (ti, yi) in enumerate(zip(t_vec, y_vec.T)):
                xi = yi[:n_x]
                lami = yi[n_x:n_x + n_lam]
                ui = yi[n_x + n_lam:n_x + n_lam + n_u]

                y_dot[:n_x, idx] = state_dyn(ti, xi, ui, p, k)
                y_dot[n_x:n_x + n_lam, idx] = costate_dyn(ti, xi, lami, ui, p, k)
                y_dot[n_x + n_lam:n_x + n_lam + n_u, idx] = control_dyn(ti, xi, lami, ui, p, k)

            return y_dot * tau_mult

        if self.use_jit_compile:
            dynamics = jit_compile(dynamics, (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray))

        return dynamics

    def _generate_ocp_diff_bcs(self, dual_ocp: CompDualOCP) -> _bc_type:
        ocp_bc0 = dual_ocp.comp_ocp.boundary_conditions.initial
        ocp_bcf = dual_ocp.comp_ocp.boundary_conditions.terminal

        dual_bc0 = dual_ocp.comp_dual.adjoined_boundary_conditions.initial
        dual_bcf = dual_ocp.comp_dual.adjoined_boundary_conditions.terminal

        control_bc = dual_ocp.control_handler.control_bc

        n_x = dual_ocp.comp_ocp.num_states
        n_u = dual_ocp.comp_ocp.num_controls
        n_p = dual_ocp.comp_ocp.num_parameters
        n_lam = dual_ocp.comp_dual.num_costates
        n_nu_0 = dual_ocp.comp_dual.num_initial_adjoints
        n_nu_f = dual_ocp.comp_dual.num_terminal_adjoints

        def boundary_conditions(y0: NPArray, yf: NPArray, _p: NPArray, k: NPArray):
            x0 = y0[:n_x]
            lam0 = y0[n_x:n_x + n_lam]
            u0 = y0[n_x + n_lam:n_x + n_lam + n_u]

            xf = yf[:n_x]
            lamf = yf[n_x:n_x + n_lam]
            uf = yf[n_x + n_lam:n_x + n_lam + n_u]

            p = _p[:n_p]
            nu0 = _p[n_p:n_p + n_nu_0]
            nuf = _p[n_p + n_nu_0: n_p + n_nu_0 + n_nu_f]
            t0, tf = _p[-2], _p[-1]

            return np.concatenate((
                np.asarray(ocp_bc0(t0, x0, u0, p, k)), np.asarray(ocp_bcf(tf, xf, uf, p, k)),
                np.asarray(dual_bc0(t0, x0, lam0, u0, p, nu0, k)), np.asarray(dual_bcf(tf, xf, lamf, uf, p, nuf, k)),
                np.asarray(control_bc(t0, x0, lam0, u0, p, k))
            ))

        if self.use_jit_compile:
            boundary_conditions = jit_compile(boundary_conditions, (NumbaArray, NumbaArray, NumbaArray, NumbaArray))

        return boundary_conditions

    @staticmethod
    def _preprocess_ocp_diff_sol(sol: Solution) -> tuple[NPArray, NPArray, NPArray]:
        t0, tf = sol.t[0], sol.t[-1]
        tau_guess = (sol.t - t0) / (tf - t0)
        p_guess = np.concatenate((sol.p, sol.nu0, sol.nuf, np.array([t0, tf])))
        y = np.vstack((sol.x, sol.lam, sol.u))
        return tau_guess, y, p_guess

    @staticmethod
    def _generate_postprocess_ocp_diff_sol(dual_ocp: CompDualOCP) -> Callable[[_scipy_bvp_sol, ndarray], Solution]:

        n_x = dual_ocp.comp_ocp.num_states
        n_p = dual_ocp.comp_ocp.num_parameters
        n_u = dual_ocp.comp_ocp.num_controls

        n_lam = dual_ocp.comp_dual.num_costates
        n_nu_0 = dual_ocp.comp_dual.num_initial_adjoints
        n_nu_f = dual_ocp.comp_dual.num_terminal_adjoints

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

    def solve(self, constants: NPArray, guess: Solution)-> Solution:
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

        tau_guess, x_guess, p_guess = self.preprocess(guess)
        k = constants

        sol: _scipy_bvp_sol = solve_bvp(
                lambda tau, x, p: self.dynamics(tau, x, p, k),
                lambda x0, xf, p: self.boundary_conditions(x0, xf, p, k),
                tau_guess, x_guess, p_guess,
                tol=self.tol, bc_tol=self.bc_tol, max_nodes=self.max_nodes, verbose=self.verbose
        )

        return self.postprocess(sol, k)
