import numpy as np

import giuseppe
from giuseppe.continuation import ContinuationHandler, SolutionSet
from giuseppe.guess_generators import generate_single_constant_guess
from giuseppe.io import InputOCP
from giuseppe.numeric_solvers.bvp import ScipySolveBVP
from giuseppe.problems.dual import SymDual, SymDualOCP, CompDualOCP
from giuseppe.problems.ocp import SymOCP
from giuseppe.guess_generators.projection import project_to_nullspace, match_constants_to_bcs
from giuseppe.guess_generators.constant.simple import update_value_constant
from giuseppe.utils import Timer

giuseppe.utils.complilation.JIT_COMPILE = True

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


with Timer(prefix='Complilation Time:'):
    sym_ocp = SymOCP(ocp)
    sym_dual = SymDual(sym_ocp)
    sym_bvp = SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
    sym_bvp.control_handler.control_law.pop(0)
    comp_dual_ocp = CompDualOCP(sym_bvp)
    num_solver = ScipySolveBVP(comp_dual_ocp)

guess = generate_single_constant_guess(comp_dual_ocp)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = SolutionSet(sym_bvp, seed_sol)
cont = ContinuationHandler(sol_set)
cont.add_linear_series(5, {'x_f': 30, 'y_f': -30}, bisection=True)

with Timer(prefix='Continuation Time:'):
    for series in cont.continuation_series:
        for k, last_sol in series:
            sol_i = num_solver.solve(k, last_sol)
            sol_set.append(sol_i)

sol = sol_set[-1]
guess = generate_single_constant_guess(comp_dual_ocp, constant=0.1)

t = guess.t
x = guess.x
u = guess.u
p = guess.p
k = guess.k
lam = guess.lam
nu0 = guess.nu0
nuf = guess.nuf

psi_0 = comp_dual_ocp.comp_ocp.boundary_conditions.initial
psi_f = comp_dual_ocp.comp_ocp.boundary_conditions.terminal

x0_star = project_to_nullspace(lambda x0: np.asarray(psi_0(t[0], x0, u[:, 0], p, k)), x[:, 0])
xf_star = project_to_nullspace(lambda xf: np.asarray(psi_f(t[-1], xf, u[:, -1], p, k)), x[:, -1])

k_star = match_constants_to_bcs(comp_dual_ocp.comp_ocp, guess)
