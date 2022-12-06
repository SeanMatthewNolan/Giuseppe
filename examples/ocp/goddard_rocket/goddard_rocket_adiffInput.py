import casadi as ca
import numpy as np
import pickle
import giuseppe

ENFORCE_CONSTRAINTS = True

goddard = giuseppe.io.AdiffInputOCP()

# Constants
max_thrust = ca.SX.sym('max_thrust', 1)
g = ca.SX.sym('g', 1)
sigma = ca.SX.sym('sigma', 1)
c = ca.SX.sym('c', 1)
h_ref = ca.SX.sym('h_ref', 1)

goddard.add_constant(max_thrust, 193.044)
goddard.add_constant(g, 32.174)
goddard.add_constant(sigma, 5.49153484923381010e-5)
goddard.add_constant(c, 1580.9425279876559)
goddard.add_constant(h_ref, 23_800)

# Independent variable
t = ca.SX.sym('t', 1)
if ENFORCE_CONSTRAINTS:
    goddard.set_independent(t, increasing=True, lower_bound=0)
else:
    goddard.set_independent(t)

# Control
thrust = ca.SX.sym('thrust', 1)
if ENFORCE_CONSTRAINTS:
    # goddard.add_control(thrust)
    goddard.add_control(thrust, lower_bound=0, upper_bound=max_thrust)
else:
    goddard.add_control(thrust)

# States
h = ca.SX.sym('h', 1)
v = ca.SX.sym('v', 1)
m = ca.SX.sym('m', 1)

# Boundary Values
h_0 = ca.SX.sym('h_0', 1)
v_0 = ca.SX.sym('v_0', 1)
m_0 = ca.SX.sym('m_0', 1)
m_f = ca.SX.sym('m_f', 1)
eps_thrust = ca.SX.sym('eps_thrust')

goddard.add_constant(h_0, 0)
goddard.add_constant(v_0, 0)
goddard.add_constant(m_0, 3)
goddard.add_constant(m_f, 1)
# goddard.add_constant(m_f, 2.95)
goddard.add_constant(eps_thrust, 0.01)
# goddard.add_constant(eps_h, 1e-3)

# Equations of Motion
goddard.add_state(h, v)
goddard.add_state(v, (thrust - sigma * v**2 * ca.exp(-h / h_ref))/m - g)
if ENFORCE_CONSTRAINTS:
    goddard.add_state(m, -thrust / c, lower_bound=m_f, upper_bound=m_0)
else:
    goddard.add_state(m, -thrust / c)

goddard.set_cost(0, eps_thrust * thrust**2, -h)

goddard.add_constraint('initial', t)
goddard.add_constraint('initial', h - h_0)
goddard.add_constraint('initial', v - v_0)
goddard.add_constraint('initial', m - m_0)

goddard.add_constraint('terminal', m - m_f)

# goddard.add_inequality_constraint(
#         'control', thrust, lower_limit=0, upper_limit=max_thrust,
#         regularizer=giuseppe.regularization.AdiffControlConstraintHandler(eps_thrust * h_ref, method='sin'))

goddard.add_inequality_constraint(
        'path', thrust, lower_limit=0, upper_limit=max_thrust,
        regularizer=giuseppe.regularization.AdiffPenaltyConstraintHandler(eps_thrust * h_ref, method='utm'))

# goddard.add_inequality_constraint(
#     'path', h, lower_limit=-10, upper_limit=1_000_000,
#     regularizer=giuseppe.regularization.AdiffPenaltyConstraintHandler(eps_h, method='utm')
# )

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_ocp = giuseppe.problems.AdiffOCP(goddard)
    adiff_dual = giuseppe.problems.AdiffDual(adiff_ocp)
    adiff_dualocp = giuseppe.problems.AdiffDualOCP(adiff_ocp, adiff_dual)
    comp_dualocp = giuseppe.problems.CompDualOCP(adiff_dualocp)
    num_solver = giuseppe.numeric_solvers.AdiffPythonSolveBVP(adiff_dualocp, verbose=2)
    # num_solver = giuseppe.numeric_solvers.ScipySolveBVP(comp_dualocp, verbose=False)

# Initialize guess with max thrust
guess = giuseppe.guess_generators.auto_propagate_guess(adiff_dualocp,
                                                       control=np.asarray(adiff_ocp.ca_control2pseudo(
                                                           193, adiff_ocp.default_values)), t_span=10)
# guess = giuseppe.guess_generators.auto_linear_guess(adiff_dualocp)
# guess = giuseppe.guess_generators.generate_constant_guess(adiff_dualocp)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = giuseppe.io.SolutionSet(adiff_dualocp, seed_sol)

cont = giuseppe.continuation.ContinuationHandler(sol_set)
# cont.add_linear_series(1, {'m_f': 1})
cont.add_logarithmic_series(2, {'eps_thrust': 1e-4})
cont.add_linear_series(1, {'max_thrust': 189.5})
sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')

with open('bvp_sols.data', 'wb') as file:
    pickle.dump(num_solver.solution_sets, file)
