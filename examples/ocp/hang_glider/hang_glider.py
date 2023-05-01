import numpy as np
import giuseppe
import pickle

glider = giuseppe.problems.input.StrInputProb()

# Dynamics
glider.set_independent('t')

glider.add_state('x', 'vx')
glider.add_state('y', 'vy')
glider.add_state('vx', '(lift * sin_alpha - drag * cos_alpha) / mass')
glider.add_state('vy', '(lift * cos_alpha + drag * sin_alpha) / mass - g')

glider.add_expression('v_air', '((u_air - vy)**2 + vx**2) ** 0.5')
glider.add_expression('sin_alpha', '(u_air - vy) / v_air')
glider.add_expression('cos_alpha', 'vx / v_air')
glider.add_expression('u_air', 'u_thermal_max * (1 - thermal_dist ** 2) * exp(-thermal_dist ** 2)')
glider.add_expression('thermal_dist', '(x - center_thermal) / radius_thermal')

glider.add_constant('u_thermal_max', 2.5)
glider.add_constant('radius_thermal', 100)
glider.add_constant('center_thermal', 2.5 * 100)

glider.add_expression('qdyn', '0.5 * rho * v_air ** 2')
glider.add_expression('lift', 'qdyn * s_ref * CL')
glider.add_expression('drag', 'qdyn * s_ref * (CD0 + CD1 * CL**2)')

cd0 = 0.034
cd1 = 0.069662
glider.add_constant('s_ref', 14)
glider.add_constant('rho', 1.13)
glider.add_constant('CD0', cd0)
glider.add_constant('CD1', cd1)

glider.add_constant('mass', 100)
glider.add_constant('g', 9.80665)

# Bounded Control
glider.add_control('CL')
glider.add_constant('eps_CL', 1e-1)
glider.add_constant('CL_min', 0.)
glider.add_constant('CL_max', 1.4)

glider.add_inequality_constraint(
    'path', 'CL', 'CL_min', 'CL_max',
    regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler('eps_CL', 'utm')
)

# Boundary Conditions
x0 = 0.
glider.add_constant('x0', 300)
glider.add_constant('y0', 1e3)
glider.add_constant('vx0', 13.227567500)
glider.add_constant('vy0', -1.2875005200)

glider.add_constant('pos_scale', 1e3)
glider.add_constant('v_scale', 13.227567500)

glider.add_constraint('initial', 't')
glider.add_constraint('initial', '(x - x0) / pos_scale')
glider.add_constraint('initial', '(y - y0) / pos_scale')
glider.add_constraint('initial', '(vx - vx0) / v_scale')
glider.add_constraint('initial', '(vy - vy0) / v_scale')

yf = 900.
vxf = 13.227567500
vyf = -1.2875005200

glider.add_constant('yf', yf)
glider.add_constant('vxf', vxf)
glider.add_constant('vyf', vyf)

glider.add_constraint('terminal', '(y - yf) / pos_scale')
glider.add_constraint('terminal', '(vx - vxf) / v_scale')
glider.add_constraint('terminal', '(vy - vyf) / v_scale')

# Cost (max range)
glider.set_cost('0', '0', '-x / pos_scale')

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_robot = giuseppe.problems.symbolic.SymDual(glider, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_robot, verbose=2, max_nodes=100, node_buffer=10, bc_tol=1e-7)


def ctrl2reg(u, min_u, max_u) -> np.array:
    return np.arcsin((2*u - min_u - max_u) / (max_u - min_u))


def reg2ctrl(u_reg, min_u, max_u) -> np.array:
    return 0.5 * ((max_u - min_u) * np.sin(u_reg) + max_u + min_u)


guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_robot,
    control=np.array((ctrl2reg(1.0, 0., 1.4),)),
    t_span=1.0)

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)

cont.add_linear_series(100, {'yf': yf, 'vxf': vxf, 'vyf': vyf})
cont.add_linear_series(100, {'x0': x0})
cont.add_logarithmic_series(100, {'eps_CL': 1e-10})

sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
