import pickle

from giuseppe.continuation import ContinuationHandler, SolutionSet
from giuseppe.guess_generators import initialize_guess_w_default_value
from giuseppe.io import InputOCP
from giuseppe.numeric_solvers.bvp import ScipySolveBVP
from giuseppe.problems.dual import SymDual, SymDualOCP, CompDualOCP
from giuseppe.problems.ocp import SymOCP
from giuseppe.utils import Timer

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

guess = initialize_guess_w_default_value(comp_dual_ocp)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = SolutionSet(sym_bvp, seed_sol)
cont = ContinuationHandler(sol_set)
cont.add_linear_series(5, {'x_f': 30, 'y_f': -30}, bisection=True)

with Timer(prefix='Continuation Time:'):
    for series in cont.continuation_series:
        for k, last_sol in series:
            sol_i = num_solver.solve(k, last_sol)
            sol_set.append(sol_i)

with open('sol_set.data', 'wb') as file:
    pickle.dump(sol_set, file)
