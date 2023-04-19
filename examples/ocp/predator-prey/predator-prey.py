import os
import pickle

import giuseppe

os.chdir(os.path.dirname(__file__))  # Set directory to current location

prob = giuseppe.problems.input.StrInputProb()

prob.set_independent('t')

prob.add_state('x_1', 'alpha * x_1 - beta * x_1 * x_2')
prob.add_state('x_2', 'delta * x_1 * x_2 - (gamma + l * u) * x_2')

prob.add_control('u')

prob.add_constant('alpha', 2 / 3)
prob.add_constant('beta', 4 / 3)
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
#                                regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler('eps'))

prob.add_inequality_constraint(
        'control', 'u', lower_limit='0', upper_limit='u_max',
        regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps', 'atan'))

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    sym_prob = giuseppe.problems.symbolic.SymDual(prob)
    num_solver = giuseppe.numeric_solvers.SciPySolver(sym_prob)

# guess = giuseppe.guess_generation.auto_propagate_guess(comp_dual, control=0.5, t_span=1)
guess = giuseppe.guess_generation.auto_propagate_guess(sym_prob, control=0, t_span=0.1)

cont = giuseppe.continuation.ContinuationHandler(num_solver, guess)
cont.add_linear_series(10, {'t_f': 1})
cont.add_linear_series(100, {'t_f': 5})
cont.add_linear_series(100, {'t_f': 7})
cont.add_logarithmic_series(30, {'eps': 1e-3})
sol_set = cont.run_continuation()

with open('sol_set.data', 'wb') as file:
    pickle.dump(sol_set, file)
