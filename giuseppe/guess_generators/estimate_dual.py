from typing import Union

from ..problems.ocp import CompOCP, OCPSol
from ..problems.dual import CompDualOCP, DualOCPSol
from ..utils.numerical_derivatives import central_difference as diff, central_difference_jacobian as jac


def estimate_dual(comp_prob: Union[CompOCP, CompDualOCP], ocp_guess: OCPSol):
    if isinstance(comp_prob, CompDualOCP):
        comp_prob = comp_prob.comp_ocp

    t = ocp_guess.t
    x = ocp_guess.x
    u = ocp_guess.u
    p = ocp_guess.p
    k = ocp_guess.k

    t0, x0, u0 = t[0], x[:, 0], u[:, 0]
    tf, xf, uf = t[-1], x[:, -1], u[:, -1]

    phi_0 = comp_prob.cost.initial
    path = comp_prob.cost.path
    phi_f = comp_prob.cost.terminal

    psi_0 = comp_prob.boundary_conditions.initial
    f = comp_prob.dynamics
    psi_f = comp_prob.boundary_conditions.terminal

    # nu_0 = diff(lambda _t: phi_0(_t, x0, u0, p, k), t0) @ diff(lambda _t: psi_0(_t, x0, u0, p, k), t0) \
    #     + jac(lambda _x: phi_0(t0, _x, u0, p, k), x0) @ jac(lambda _x: psi_0(t0, _x, u0, p, k), x0) \
    #     + jac(lambda _u: phi_0(t0, x0, _u, p, k), u0) @ jac(lambda _u: psi_0(t0, x0, _u, p, k), u0) \
    #     + jac(lambda _p: phi_0(t0, x0, u0, _p, k), p) @ jac(lambda _p: psi_0(t0, x0, u0, _p, k), p)

    raise NotImplementedError
