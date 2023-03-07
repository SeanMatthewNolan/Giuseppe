import numpy as np
import giuseppe
import pickle

robot = giuseppe.problems.input.StrInputProb()

robot.set_independent('t')

robot.add_state('x', 'vx')
robot.add_state('y', 'vy')
robot.add_state('vx', '(T1 + T2) * cos(tha)')
robot.add_state('vy', '(T1 + T2) * sin(tha)')
robot.add_state('tha', 'om')
robot.add_state('om', 'alp * T1 - bet * T2')

robot.add_control('u1')
robot.add_control('u2')
robot.add_control('u3')
robot.add_control('u4')

robot.add_expression('T1', 'u1 - u2')
robot.add_expression('T2', 'u3 - u4')

alp = 0.2
bet = 0.2
robot.add_constant('alp', alp)
robot.add_constant('bet', bet)

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

min_u = 0.0
max_u = 10.0
robot.add_constant('eps_u', 5e-1)
robot.add_constant('min_u', min_u)
robot.add_constant('max_u', max_u)

min_T = -1.0
max_T = 1.0
robot.add_constant('eps_T', 1e-1)
robot.add_constant('min_T', min_T)
robot.add_constant('max_T', max_T)

robot.set_cost('0', 'u1 + u2 + u3 + u4', '0')

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
        'control', 'u1', lower_limit='min_u', upper_limit='max_u',
        regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method='sin'))

robot.add_inequality_constraint(
        'control', 'u2', lower_limit='min_u', upper_limit='max_u',
        regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method='sin'))

robot.add_inequality_constraint(
        'control', 'u3', lower_limit='min_u', upper_limit='max_u',
        regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method='sin'))

robot.add_inequality_constraint(
        'control', 'u4', lower_limit='min_u', upper_limit='max_u',
        regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method='sin'))

robot.add_inequality_constraint(
        'path', 'T1', lower_limit='min_T', upper_limit='max_T',
        regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler('eps_T', method='utm'))

robot.add_inequality_constraint(
        'path', 'T2', lower_limit='min_T', upper_limit='max_T',
        regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler('eps_T', method='utm'))


with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_robot = giuseppe.problems.symbolic.SymDual(robot, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_robot, verbose=0, max_nodes=0, node_buffer=10)


def ctrl2reg(u: np.array) -> np.array:
    # return eps_u * np.tan((2 * u - max_u - min_u) * np.pi / (2 * (max_u - min_u)))
    # return u
    return np.arcsin((2*u - min_u - max_u) / (max_u - min_u))


def reg2ctrl(u_reg: np.array) -> np.array:
    return 0.5 * ((max_u - min_u) * np.sin(u_reg) + max_u + min_u)


guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_robot,
    control=ctrl2reg(np.array((0.0, 0.5, 0.5, 0.0))),
    t_span=1.0)

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)

cont.add_linear_series(100, {'t_f': t_f})
cont.add_linear_series(100, {'x_f': x_f, 'vx_f': vx_f})
cont.add_linear_series(100, {'y_f': y_f, 'vy_f': vy_f})
cont.add_linear_series(100, {'tha_f': tha_f, 'om_f': om_f})
cont.add_logarithmic_series(100, {'eps_T': 1e-2, 'eps_u': 5e-2})
cont.add_logarithmic_series(150, {'eps_T': 1e-3, 'eps_u': 5e-3})
cont.add_logarithmic_series(350, {'eps_T': 1e-4, 'eps_u': 5e-4})

sol_set = cont.run_continuation()

sol_set.save('sol_set.data')

# # Initial guess: rotate 90 deg
# ang_acc = 2 * (tha_f - tha_0) / (t_f ** 2)
# t2 = -ang_acc / bet
# guess = giuseppe.guess_generators.auto_propagate_guess(comp_dual_ocp,
#                                                        control=ctrl2reg(np.array((0.0, 0.0, t2, 0.0))),
#                                                        t_span=1.0)

#
# seed_sol = num_solver.solve(guess.k, guess)
#
# with open('seed_sol.data', 'wb') as f:
#     pickle.dump(seed_sol, f)
#
# sol_set = giuseppe.io.SolutionSet(sym_bvp, seed_sol)
# x_f_seed = seed_sol.x[:, -1]
# v_f_intermediate = 0.1 * (vy_f + x_f_seed[3])
#
# cont = giuseppe.continuation.ContinuationHandler(sol_set)
# cont.add_linear_series(100, {'y_f': y_f, 'vy_f': 0.1 * (vy_f + x_f_seed[3])})
# # cont.add_linear_series(100, {'x_f': x_f, 'vx_f': 0.1 * (vy_f + x_f_seed[3]),
# #                              'vy_f': vy_f, 'tha_f': tha_f})
# sol_set = cont.run_continuation(num_solver)
#
# # cont.add_linear_series(50, {'x_f': x_f, 'y_f': y_f,
# #                            'vx_f': vx_f, 'vy_f': vy_f,
# #                            'tha_f': tha_f, 'om_f': om_f}, bisection=True)
# # sol_set = cont.run_continuation(num_solver)
# # cont.add_linear_series(10, {'m_f': 1})
# # cont.add_logarithmic_series(20, {'eps_thrust': 1e-6})
# # sol_set = cont.run_continuation(num_solver)
#
# # sol_set.save('sol_set.data')
