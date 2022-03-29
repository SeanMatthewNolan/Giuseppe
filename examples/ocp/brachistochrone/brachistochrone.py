import numpy as np

import giuseppe
from giuseppe.io import InputOCP
from giuseppe.problems.dual import SymDual, SymDualOCP, CompDual, CompDualOCP
from giuseppe.problems.ocp import SymOCP, CompOCP

ocp = InputOCP()

ocp.set_independent('t')

ocp.add_state('x', 'v*cos(theta)')
ocp.add_state('y', 'v*sin(theta)')
ocp.add_state('v', '-g*sin(theta)')

ocp.add_control('theta')

ocp.add_constant('g', 32.2)

ocp.add_constant('x_0')
ocp.add_constant('y_0')
ocp.add_constant('v_0')

ocp.add_constant('x_f')
ocp.add_constant('y_f')

ocp.set_cost('0', '1', '0')

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'x - x_0')
ocp.add_constraint('initial', 'y - y_0')
ocp.add_constraint('initial', 'v - v_0')

ocp.add_constraint('terminal', 'x - x_f')
ocp.add_constraint('terminal', 'y - y_f')

sym_ocp = SymOCP(ocp)
sym_dual = SymDual(sym_ocp)
sym_bvp_alg = SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
sym_bvp_dif = SymDualOCP(sym_ocp, sym_dual, control_method='differential')

# giuseppe.utils.complilation.JIT_COMPILE = False

comp_ocp = CompOCP(sym_ocp)

t0 = 0.
x0 = np.array([0., 0., 1.])
u0 = np.array([-np.pi / 4])

tf = 10.
xf = np.array([12., -12, 12])
uf = np.array([-np.pi / 6])

k = sym_ocp.default_values

f0 = comp_ocp.dynamics(t0, x0, u0, k)

psi0 = comp_ocp.boundary_conditions.initial(t0, x0, k)
psif = comp_ocp.boundary_conditions.terminal(tf, xf, k)

phi0 = comp_ocp.cost.initial(t0, x0, k)
ll = comp_ocp.cost.path(t0, x0, u0, k)
phif = comp_ocp.cost.terminal(tf, xf, k)

lam0 = np.array([-0.1, -0.2, -0.3])
lamf = np.array([-0.1, -0.2, -0.3])

nu0 = np.array([0.01, 0.02, 0.03, 0.04])
nuf = np.array([-0.01, -0.02])

comp_dual = CompDual(sym_dual)

lam_dot0 = comp_dual.costate_dynamics(t0, x0, lam0, u0, k)

adj_bc0 = comp_dual.adjoined_boundary_conditions.initial(t0, x0, lam0, u0, nu0, k)
adj_bcf = comp_dual.adjoined_boundary_conditions.terminal(tf, xf, lamf, uf, nuf, k)

aug_cost0 = comp_dual.augmented_cost.initial(t0, x0, lam0, u0, nu0, k)
aug_costf = comp_dual.augmented_cost.terminal(tf, xf, lamf, uf, nuf, k)

ham0 = comp_dual.hamiltonian(t0, x0, lam0, u0, k)

# sym_bvp_alg.control_handler.control_law.pop()
comp_dual_ocp = CompDualOCP(sym_bvp_alg)
u0 = comp_dual_ocp.control_handler.control(t0, x0, lam0, k)
