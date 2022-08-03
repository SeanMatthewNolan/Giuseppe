import pickle

import giuseppe

giuseppe.utils.complilation.JIT_COMPILE = True

prob = giuseppe.io.InputOCP()

prob.set_independent('t')

# prob.add_state('x_1', 'x_1 - x_1 * x_2')
# prob.add_state('x_2', 'x_1 * x_2 - (k + l * u) * x_2')

prob.add_state('x_1', 'alpha * x_1 - beta * x_1 * x_2')
prob.add_state('x_2', 'delta * x_1 * x_2 - (gamma + l * u) * x_2')

prob.add_control('u')

prob.add_constant('alpha', 2/3)
prob.add_constant('beta', 4/2)
prob.add_constant('gamma', 1)
prob.add_constant('delta', 1)

prob.add_constant('a', 0.01)
prob.add_constant('l', 0.5)
prob.add_constant('k', 0.1)

prob.add_constant('x_1_0', 1)
prob.add_constant('x_2_0', 1)

prob.add_constant('t_f', 100)

# prob.add_constant('eps', 1e-2)
prob.add_constant('eps', 1e-1)
prob.add_constant('u_max', 1)

prob.set_cost('0', 'a * u', '-x_1')

prob.add_constraint('initial', 't')
prob.add_constraint('initial', 'x_1 - x_1_0')
prob.add_constraint('initial', 'x_2 - x_2_0')

prob.add_constraint('terminal', 't - t_f')

# prob.add_inequality_constraint('path', 'u', lower_limit='0', upper_limit='u_max',
#                                regularizer=giuseppe.regularization.PenaltyConstraintHandler('eps'))

prob.add_inequality_constraint('control', 'u', lower_limit='0', upper_limit='u_max',
                               regularizer=giuseppe.regularization.ControlConstraintHandler('eps', 'atan'))

with giuseppe.utils.Timer(prefix='Complilation Time:'):
    sym_ocp = giuseppe.problems.SymOCP(prob)
    sym_dual = giuseppe.problems.SymDual(sym_ocp)
    sym_bvp = giuseppe.problems.SymDualOCP(sym_ocp, sym_dual, control_method='differential')
    comp_dual_ocp = giuseppe.problems.CompDualOCP(sym_bvp, use_jit_compile=True)
    num_solver = giuseppe.numeric_solvers.ScipySolveBVP(comp_dual_ocp, use_jit_compile=True)

# guess = giuseppe.guess_generators.auto_propagate_guess(comp_dual_ocp, control=0.5, t_span=1)
guess = giuseppe.guess_generators.auto_propagate_guess(comp_dual_ocp, control=0, t_span=0.1)

with open('guess.data', 'wb') as file:
    pickle.dump(guess, file)

seed_sol = num_solver.solve(guess.k, guess)
sol_set = giuseppe.io.SolutionSet(sym_bvp, seed_sol)

cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(10, {'t_f': 1}, bisection=True)
cont.add_linear_series(100, {'t_f': 5}, bisection=True)
cont.add_linear_series(100, {'t_f': 7}, bisection=True)
cont.add_logarithmic_series(30, {'eps': 1e-3}, bisection=True)
cont.run_continuation(num_solver)

with open('sol_set.data', 'wb') as file:
    pickle.dump(sol_set, file)
