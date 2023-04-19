import numpy as np
import giuseppe
import pickle

from giuseppe.problems.symbolic.regularization import ControlConstraintHandler, PenaltyConstraintHandler

robot = giuseppe.problems.input.StrInputProb()

robot.set_independent('t')

robot.add_state('x', 'vx')
robot.add_state('y', 'vy')
robot.add_state('vx', '(T1 + T2) * cos(tha)')
robot.add_state('vy', '(T1 + T2) * sin(tha)')
robot.add_state('tha', 'om')
robot.add_state('om', 'alp * T1 - bet * T2')

# robot.add_control('u1')
# robot.add_control('u2')
# robot.add_control('u3')
# robot.add_control('u4')

# robot.add_expression('T1', 'u1**2 - u2**2')
# robot.add_expression('T2', 'u3**2 - u4**2')

robot.add_control('T1')
robot.add_control('T2')

robot.add_constant('alp', 0.2)
robot.add_constant('bet', 0.2)

x_0 = -10.0
y_0 = -10.0
vx_0 = 0.0
vy_0 = 0.0
tha_0 = np.pi/2
om_0 = 0.0
robot.add_constant('x_0', x_0)
robot.add_constant('y_0', y_0)
robot.add_constant('vx_0', vx_0)
robot.add_constant('vy_0', vy_0)
robot.add_constant('tha_0', tha_0)
robot.add_constant('om_0', om_0)

t_f = 12.0
x_f = 0.0
y_f = 0.0
vx_f = 0.0
vy_f = 0.0
tha_f = 0.0
om_f = 0.0
robot.add_constant('t_f', t_f)
robot.add_constant('x_f', x_f)
robot.add_constant('y_f', y_f)
robot.add_constant('vx_f', vx_f)
robot.add_constant('vy_f', vy_f)
robot.add_constant('tha_f', tha_f)
robot.add_constant('om_f', om_f)

# min_u = 0.0
# max_u = 1000.0
# eps_u = 1e-3
# robot.add_constant('eps_u', eps_u)
# robot.add_constant('min_u', min_u)
# robot.add_constant('max_u', max_u)

min_T = -1.0
max_T = 1.0
robot.add_constant('eps_T', 1)
robot.add_constant('min_T', min_T)
robot.add_constant('max_T', max_T)

robot.set_cost('0', '(T1**2)**0.5 + (T2**2)**0.5', '0')

robot.add_constraint('initial', 't')
robot.add_constraint('initial', 'x - x_0')
robot.add_constraint('initial', 'y - y_0')
robot.add_constraint('initial', 'vx - vx_0')
robot.add_constraint('initial', 'vy - vy_0')
robot.add_constraint('initial', 'tha - tha_0')
robot.add_constraint('initial', 'om - om_0')

robot.add_constraint('terminal', 't - t_f')
robot.add_constraint('terminal', 'x - x_f')
robot.add_constraint('terminal', 'y - y_f')
robot.add_constraint('terminal', 'vx - vx_f')
robot.add_constraint('terminal', 'vy - vy_f')
robot.add_constraint('terminal', 'tha - tha_f')
robot.add_constraint('terminal', 'om - om_f')

robot.add_inequality_constraint(
        'path', 'T1', lower_limit='min_T', upper_limit='max_T',
        regularizer=PenaltyConstraintHandler('eps_T', method='utm'))

robot.add_inequality_constraint(
        'path', 'T2', lower_limit='min_T', upper_limit='max_T',
        regularizer=PenaltyConstraintHandler('eps_T', method='utm'))


with giuseppe.utils.Timer(prefix='Compilation Time:'):
    sym_dual = giuseppe.problems.symbolic.SymDual(robot, control_method='differential')
    comp_dual_ocp = sym_dual
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_dual_ocp)


guess = giuseppe.guess_generation.auto_propagate_guess(
        comp_dual_ocp, control=(0.5, 0.5), t_span=1)

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

cont = giuseppe.continuation.ContinuationHandler(num_solver, guess)
cont.add_linear_series(10, {'t_f': 12})
cont.add_linear_series(100, {'y_f': 0, 'vy_f': 0})
cont.add_linear_series(100, {'x_f': 0, 'vx_f': 0})
cont.add_linear_series(100, {'om_f': 0, 'tha_f': 0})
cont.add_logarithmic_series(100, {'eps_T': 1e-8})
sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
