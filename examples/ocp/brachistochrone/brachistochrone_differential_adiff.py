import pickle

import numpy as np

import giuseppe
from giuseppe.continuation import ContinuationHandler, SolutionSet
from giuseppe.io import InputOCP
from giuseppe.numeric_solvers.bvp import AdiffScipySolveBVP
from giuseppe.problems.dual import AdiffDual, AdiffDualOCP
from giuseppe.problems.ocp import SymOCP, AdiffOCP
# from giuseppe.guess_generators import generate_constant_guess
from giuseppe.utils import Timer

giuseppe.utils.complilation.JIT_COMPILE = True
giuseppe.utils.complilation.LAMB_MODS = ['numpy']

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

with Timer(prefix='Compilation Time:'):
    sym_ocp = SymOCP(ocp)
    adiff_ocp = AdiffOCP(sym_ocp)
    adiff_dual = AdiffDual(adiff_ocp)
    adiff_dualocp = AdiffDualOCP(adiff_ocp, adiff_dual)
    # num_solver = AdiffScipySolveBVP(adiff_dualocp)
#
# n = 2
# t = np.linspace(0, 0.25, n)
# x = np.linspace(np.array([0., 0., 1.]), np.array([1., 1., 8.]), n).T
# lam = np.linspace(np.array([-0.1, -0.1, -0.1]), np.array([-0.1, -0.1, -0.1]), n).T
# u = np.linspace(np.array([-1]), np.array([-0.5]), n).T
# nu0 = np.array([-0.1, -0.1, -0.1, -0.1])
# nuf = np.array([-0.1, -0.1])
# k = sym_ocp.default_values
#
# guess = num_solver.solve(k, DualOCPSol(t=t, x=x, lam=lam, u=u, nu0=nu0, nuf=nuf, k=k))
# sol_set = SolutionSet(sym_bvp, guess)
# cont = ContinuationHandler(sol_set)
# cont.add_linear_series(5, {'x_f': 30, 'y_f': -30}, bisection=True)
#
# with Timer(prefix='Continuation Time:'):
#     for series in cont.continuation_series:
#         for k, last_sol in series:
#             sol_i = num_solver.solve(k, last_sol)
#             sol_set.append(sol_i)
#
# with open('sol_set.data', 'wb') as file:
#     pickle.dump(sol_set, file)
