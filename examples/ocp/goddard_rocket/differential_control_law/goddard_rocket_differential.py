import os

import giuseppe

os.chdir(os.path.dirname(__file__))  # Set directory to current location

goddard = giuseppe.problems.input.StrInputProb()

goddard.set_independent('t')

goddard.add_state('h', 'v')
goddard.add_state('v', '(thrust - sigma * v**2 * exp(-h / h_ref))/m - g')
goddard.add_state('m', '-thrust/c')

goddard.add_control('thrust')

goddard.add_constant('max_thrust', 193.044)
goddard.add_constant('g', 32.174)
goddard.add_constant('sigma', 5.49153484923381010e-5)
goddard.add_constant('c', 1580.9425279876559)
goddard.add_constant('h_ref', 23_800)

goddard.add_constant('h_0', 0)
goddard.add_constant('v_0', 0)
goddard.add_constant('m_0', 3)

goddard.add_constant('m_f', 2.95)

goddard.add_constant('eps_thrust', 0.01)

goddard.set_cost('0', '0', '-h')

goddard.add_constraint('initial', 't')
goddard.add_constraint('initial', 'h - h_0')
goddard.add_constraint('initial', 'v - v_0')
goddard.add_constraint('initial', 'm - m_0')

goddard.add_constraint('terminal', 'm - m_f')

goddard.add_inequality_constraint(
        'control', 'thrust', lower_limit='0', upper_limit='max_thrust',
        regularizer=giuseppe.problems.symbolic.ControlConstraintHandler(
                'eps_thrust * h_ref', method='sin'))

with giuseppe.utils.Timer(prefix='Setup Time:'):
    compiled_problem = giuseppe.problems.symbolic.SymDual(goddard)
    num_solver = giuseppe.numeric_solvers.SciPySolver(compiled_problem, bc_tol=1e-8, tol=1e-5)
    guess = giuseppe.guess_generation.auto_propagate_guess(compiled_problem, control=89 / 180 * 3.14159)

cont = giuseppe.continuation.ContinuationHandler(num_solver, guess)
cont.add_linear_series(10, {'m_f': 1})
cont.add_logarithmic_series(10, {'eps_thrust': 2e-5})
sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
