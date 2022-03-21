import pickle

import numpy as np

from giuseppe.io import InputBVP
from giuseppe.problems.bvp import SymBVP, CompBVP, BVPSol
from giuseppe.numeric_solvers.bvp.scipy import ScipySolveBVP
from giuseppe.continuation import SolutionSet, SolutionSubset, ContinuationHandler
from giuseppe.utils import Timer

sturm_liouville = InputBVP()

sturm_liouville.set_independent('x')

sturm_liouville.add_state('y', 'yp')
sturm_liouville.add_state('yp', '-k**2 * y')
sturm_liouville.add_state('k', '0.')

# sturm_liouville.add_parameter('k') TODO: Add parameter support

sturm_liouville.add_constant('x_0', 0.)
sturm_liouville.add_constant('x_f', 1.)
sturm_liouville.add_constant('y_0', 0.)
sturm_liouville.add_constant('y_f', 0.)
sturm_liouville.add_constant('a', 1.)

sturm_liouville.add_constraint('initial', 'x - x_0')
sturm_liouville.add_constraint('initial', 'y - y_0')
sturm_liouville.add_constraint('initial', 'yp - a * k')

sturm_liouville.add_constraint('terminal', 'x - x_f')
sturm_liouville.add_constraint('terminal', 'y - y_f')

sym_bvp = SymBVP(sturm_liouville)
comp_bvp = CompBVP(sym_bvp)

num_solver = ScipySolveBVP(comp_bvp, do_jit_compile=True)

n_steps = 11

t0, tf = 0., 1.
x0, xf = np.array([1., 1., 1.]), np.array([1., -1., 1.])
tau_vec = np.linspace(0., 1., n_steps)
x_vec = np.linspace(x0, xf, n_steps).T

x_dot = num_solver.dynamics(tau_vec, x_vec, np.array([t0, tf]), sym_bvp.default_values)
bc = num_solver.boundary_conditions(x0, xf, np.array([t0, tf]), sym_bvp.default_values)

guess = BVPSol(t=tau_vec, x=x_vec, k=sym_bvp.default_values)

sol = num_solver.solve(sym_bvp.default_values, guess)

with open('sol.data', 'wb') as file:
    pickle.dump(sol, file)

sol_set = SolutionSet(sym_bvp, sol)
cont = ContinuationHandler(sol_set)
cont.add_linear_series(5, {'a': 2})

with Timer():
    sol_set.append(SolutionSubset())
    for series in cont.continuation_series:
        for k, guess in series:
            sol_i = num_solver.solve(k, guess)
            sol_set[-1].append(sol_i)

with open('sol_set.data', 'wb') as file:
    pickle.dump(sol_set[-1], file)
