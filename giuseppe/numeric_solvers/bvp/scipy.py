from typing import Union, Callable, TypeVar

import numpy as np
from scipy.integrate import solve_bvp

from ...problems.bvp import CompBVP, BVPSol
from ...problems.dual import CompDualOCP, DualSol
from ...problems.dual.compiled import CompAlgControlHandler, CompDiffControlHandler
from ...utils.complilation import jit_compile
from ...utils.mixins import Picky
from ...utils.typing import NumbaArray, NumbaMatrix, NPArray

_scipy_bvp_sol = TypeVar('_scipy_bvp_sol')
_dyn_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_bc_type = Callable[[NPArray, NPArray, NPArray, NPArray], NPArray]
_preprocess_type = Callable[[BVPSol], tuple[NPArray, NPArray, NPArray]]
_postprocess_type = Callable[[_scipy_bvp_sol, NPArray], BVPSol]


class ScipySolveBVP(Picky):
    SUPPORTED_INPUTS = Union[CompBVP, CompDualOCP]

    def __init__(self, bvp: SUPPORTED_INPUTS, do_jit_compile: bool = True,
                 tol: float = 0.001, bc_tol: float = 0.001, max_nodes: int = 1000, verbose: bool = False):
        Picky.__init__(self, bvp)

        # Options directly for scipy.solve_bvp
        self.tol: float = tol
        self.bc_tol: float = bc_tol
        self.max_nodes = max_nodes
        self.verbose = verbose

        self.do_jit_compile: bool = do_jit_compile

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
            elif type(bvp.control_handler) is CompDiffControlHandler:
                dynamics = self._generate_bvp_dynamics(bvp)
                boundary_conditions = self._generate_bvp_bcs(bvp)
                preprocess = self._preprocess_bvp_sol
                postprocess = self._postprocess_bvp_sol
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
            t_vec = tau_vec * tau_mult + (tf + t0) / 2

            x_dot = np.empty_like(x_vec)  # Need to pre-allocate for Numba
            for idx, (ti, xi) in enumerate(zip(t_vec, x_vec.T)):
                x_dot[:, idx] = bvp_dyn(ti, xi, k)

            return x_dot * tau_mult

        if self.do_jit_compile:
            dynamics = jit_compile(dynamics, (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray))

        return dynamics

    def _generate_bvp_bcs(self, bvp: CompBVP) -> _bc_type:
        bvp_bc0 = bvp.boundary_conditions.initial
        bvp_bcf = bvp.boundary_conditions.terminal

        def boundary_conditions(x0: NPArray, xf: NPArray, p: NPArray, k: NPArray):
            t0, tf = p[-2], p[-1]
            return np.concatenate((bvp_bc0(t0, x0, k), bvp_bcf(tf, xf, k)))

        if self.do_jit_compile:
            boundary_conditions = jit_compile(boundary_conditions, (NumbaArray, NumbaArray, NumbaArray, NumbaArray))

        return boundary_conditions

    @staticmethod
    def _preprocess_bvp_sol(sol: BVPSol) -> tuple[NPArray, NPArray, NPArray]:
        t0, tf = sol.t[0], sol.t[-1]
        p_guess = np.array([t0, tf])
        tau_guess = (sol.t - (t0 + tf) / 2) / (tf - t0)
        return tau_guess, sol.x, p_guess

    @staticmethod
    def _postprocess_bvp_sol(scipy_sol: _scipy_bvp_sol, k: NPArray) -> BVPSol:
        tau: NPArray = scipy_sol.x
        x: NPArray = scipy_sol.y
        p: NPArray = scipy_sol.p

        t0, tf = p[-2], p[-1]
        t = (tf - t0) * tau + (t0 + tf) / 2
        p = p[:-2]

        return BVPSol(t=t, x=x, p=p, k=k, converged=scipy_sol.success)

    def _generate_ocp_alg_dynamics(self, dual_ocp: CompDualOCP) -> _dyn_type:
        state_dyn = dual_ocp.comp_ocp.dynamics
        costate_dyn = dual_ocp.comp_dual.costate_dynamics

        n_x = dual_ocp.comp_ocp.num_states

        control = dual_ocp.control_handler.control

        def dynamics(tau_vec: NPArray, y_vec: NPArray, p: NPArray, k: NPArray) -> NPArray:
            t0, tf = p[-2], p[-1]
            tau_mult = (tf - t0)
            t_vec = tau_vec * tau_mult + (tf + t0) / 2

            y_dot = np.empty_like(y_vec)  # Need to pre-allocate for Numba
            for idx, (ti, yi) in enumerate(zip(t_vec, y_vec.T)):
                xi = yi[:n_x]
                lami = yi[n_x:]
                ui = control(t0, xi, lami, k)

                y_dot[:n_x, idx] = state_dyn(ti, xi, ui, k)
                y_dot[n_x:, idx] = costate_dyn(ti, xi, lami, ui, k)

            return y_dot * tau_mult

        if self.do_jit_compile:
            dynamics = jit_compile(dynamics, (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray))

        return dynamics

    def _generate_ocp_alg_bcs(self, dual_ocp: CompDualOCP) -> _bc_type:
        ocp_bc0 = dual_ocp.comp_ocp.boundary_conditions.initial
        ocp_bcf = dual_ocp.comp_ocp.boundary_conditions.terminal

        dual_bc0 = dual_ocp.comp_dual.adjoined_boundary_conditions.initial
        dual_bcf = dual_ocp.comp_dual.adjoined_boundary_conditions.terminal

        n_x = dual_ocp.comp_ocp.num_states
        n_p = 0
        n_nu_0 = dual_ocp.comp_dual.num_initial_adjoints
        n_nu_f = dual_ocp.comp_dual.num_terminal_adjoints

        control = dual_ocp.control_handler.control

        def boundary_conditions(y0: NPArray, yf: NPArray, r: NPArray, k: NPArray):
            x0 = y0[:n_x]
            lam0 = y0[n_x:]

            xf = yf[:n_x]
            lamf = yf[n_x:]

            p = r[:n_p]
            nu0 = r[n_p:n_p + n_nu_0]
            nuf = r[n_p + n_nu_0: n_p + n_nu_0 + n_nu_f]
            t0, tf = p[-2], p[-1]

            u0 = control(t0, x0, lam0, k)
            uf = control(tf, xf, lamf, k)

            return np.concatenate((
                ocp_bc0(t0, x0, k), ocp_bcf(tf, xf, k),
                dual_bc0(t0, x0, lam0, u0, nu0, k), dual_bcf(tf, xf, lamf, uf, nuf, k)
            ))

        if self.do_jit_compile:
            boundary_conditions = jit_compile(boundary_conditions, (NumbaArray, NumbaArray, NumbaArray, NumbaArray))

        return boundary_conditions

    @staticmethod
    def _preprocess_ocp_alg_sol(sol: DualSol) -> tuple[NPArray, NPArray, NPArray]:
        t0, tf = sol.t[0], sol.t[-1]
        tau_guess = (sol.t - (t0 + tf) / 2) / (tf - t0)
        p_guess = np.concatenate(sol.p, sol.nu0, sol.nuf, np.array([t0, tf]))
        y = np.vstack((sol.x, sol.lam))
        return tau_guess, y, p_guess

    @staticmethod
    def _generate_postprocess_ocp_alg_sol(dual_ocp: CompDualOCP) \
            -> Callable[[DualSol], tuple[NPArray, NPArray, NPArray]]:

        n_x = dual_ocp.comp_ocp.num_states
        n_p = 0
        n_nu_0 = dual_ocp.comp_dual.num_initial_adjoints
        n_nu_f = dual_ocp.comp_dual.num_terminal_adjoints

        control = dual_ocp.control_handler.control

        def _postprocess_ocp_alg_sol(scipy_sol: _scipy_bvp_sol, k: NPArray) -> DualSol:
            tau: NPArray = scipy_sol.x
            t0, tf = scipy_sol.p[-2], scipy_sol.p[-1]
            t: NPArray = (tf - t0) * tau + (t0 + tf) / 2

            x: NPArray = scipy_sol.y[:n_x]
            lam: NPArray = scipy_sol.y[n_x:]

            p: NPArray = scipy_sol.p[:n_p]
            nu_0: NPArray = scipy_sol.p[n_p:n_p + n_nu_0]
            nu_f: NPArray = scipy_sol.p[n_p + n_nu_0:n_p + n_nu_0 + n_nu_f]

            u = np.array([control(ti, xi, lami, k) for ti, xi, lami in zip(t, x, lam)])

            return DualSol(t=t, x=x, lam=lam, u=u, p=p, nu0=nu_0, nuf=nu_f, k=k, converged=scipy_sol.success)

        return _postprocess_ocp_alg_sol

    def solve(self, constants: NPArray, guess: Union[BVPSol]) -> Union[BVPSol]:

        tau_guess, x_guess, p_guess = self.preprocess(guess)
        k = constants

        sol: _scipy_bvp_sol = solve_bvp(
                lambda tau, x, p: self.dynamics(tau, x, p, k),
                lambda x0, xf, p: self.boundary_conditions(x0, xf, p, k),
                tau_guess, x_guess, p_guess,
                tol=self.tol, bc_tol=self.bc_tol, max_nodes=self.max_nodes, verbose=self.verbose
        )

        return self.postprocess(sol, k)
