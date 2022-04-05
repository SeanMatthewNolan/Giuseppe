import numpy as np

from giuseppe.io import InputOCP
from giuseppe.problems.dual import SymDual, SymDualOCP, CompDualOCP, DualSol
from giuseppe.problems.ocp import SymOCP
from giuseppe.numeric_solvers.bvp import ScipySolveBVP

ocp = InputOCP()

ocp.set_independent('t')

ocp.add_state('x', 'v*cos(theta)')
ocp.add_state('y', 'v*sin(theta)')
ocp.add_state('v', '-g*sin(theta)')

ocp.add_control('theta')

ocp.add_constant('g', 32.2)

ocp.add_constant('x_0', 0)
ocp.add_constant('y_0', 0)
ocp.add_constant('v_0', 1)

ocp.add_constant('x_f', 1)
ocp.add_constant('y_f', -1)

ocp.set_cost('0', '1', '0')

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'x - x_0')
ocp.add_constraint('initial', 'y - y_0')
ocp.add_constraint('initial', 'v - v_0')

ocp.add_constraint('terminal', 'x - x_f')
ocp.add_constraint('terminal', 'y - y_f')

sym_ocp = SymOCP(ocp)
sym_dual = SymDual(sym_ocp)
sym_bvp = SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
comp_dual_ocp = CompDualOCP(sym_bvp)

solver_alg = ScipySolveBVP(comp_dual_ocp)

n = 3
t = np.linspace(0, 0.25, n)
x = np.linspace(np.array([0., 0., 1.]), np.array([1., 1., 8.]), n)
lam = np.linspace(np.array([-0.1, -0.1, -0.1]), np.array([-0.1, -0.1, -0.1]), n)
nu0 = np.array([-0.1, -0.1, -0.1, -0.1])
nuf = np.array([-0.1, -0.1])
k = sym_ocp.default_values

guess = DualSol(t=t, x=x, lam=lam, nu0=nu0, nuf=nuf, k=k)

sol = solver_alg.solve(k, guess)
