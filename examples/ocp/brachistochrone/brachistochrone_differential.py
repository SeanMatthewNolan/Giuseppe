import numpy as np

from giuseppe.continuation import ContinuationHandler
from giuseppe.guess_generators import generate_constant_guess
from giuseppe.io import InputOCP, SolutionSet
from giuseppe.numeric_solvers.bvp import ScipySolveBVP
from giuseppe.problems.dual import SymDual, SymDualOCP, CompDualOCP
from giuseppe.problems.ocp import SymOCP
from giuseppe.utils import Timer

input_ocp = InputOCP()

input_ocp.set_independent('t')

input_ocp.add_state('x', 'v*cos(θ)')
input_ocp.add_state('y', 'v*sin(θ)')
input_ocp.add_state('v', '-g*sin(θ)')

input_ocp.add_control('θ')

input_ocp.add_constant('g', 32.2)

input_ocp.add_constant('x_0', 0)
input_ocp.add_constant('y_0', 0)
input_ocp.add_constant('v_0', 1)

input_ocp.add_constant('x_f', 1)
input_ocp.add_constant('y_f', -1)

input_ocp.set_cost('0', '0', 't')

input_ocp.add_constraint('initial', 't')
input_ocp.add_constraint('initial', 'x - x_0')
input_ocp.add_constraint('initial', 'y - y_0')
input_ocp.add_constraint('initial', 'v - v_0')

input_ocp.add_constraint('terminal', 'x - x_f')
input_ocp.add_constraint('terminal', 'y - y_f')

with Timer(prefix='Complilation Time:'):
    sym_ocp = SymOCP(input_ocp)
    sym_dual = SymDual(sym_ocp)
    sym_bvp = SymDualOCP(sym_ocp, sym_dual, control_method='differential')
    comp_dual_ocp = CompDualOCP(sym_bvp)
    num_solver = ScipySolveBVP(comp_dual_ocp)

guess = generate_constant_guess(
        comp_dual_ocp, t_span=0.25, x=np.array([0., 0., 1.]), lam=-0.1, u=-0.5, nu_0=-0.1, nu_f=-0.1)

seed_sol = num_solver.solve(guess.k, guess)
sol_set = SolutionSet(sym_bvp, seed_sol)
cont = ContinuationHandler(sol_set)
cont.add_linear_series(5, {'x_f': 30, 'y_f': -30}, bisection=True)

sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')

