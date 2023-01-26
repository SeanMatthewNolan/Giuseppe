import casadi as ca
import giuseppe

goddard = giuseppe.problems.input.AdiffInputProb()

# Constants
max_thrust = ca.MX.sym('max_thrust', 1)
g = ca.MX.sym('g', 1)
sigma = ca.MX.sym('sigma', 1)
c = ca.MX.sym('c', 1)
h_ref = ca.MX.sym('h_ref', 1)

goddard.add_constant(max_thrust, 193.044)  # TODO can this field be removed?
goddard.add_constant(g, 32.174)
goddard.add_constant(sigma, 5.49153484923381010e-5)
goddard.add_constant(c, 1580.9425279876559)
goddard.add_constant(h_ref, 23_800)

# Independent variable
t = ca.MX.sym('t', 1)
goddard.set_independent(t)

# Control
thrust = ca.MX.sym('thrust', 1)
goddard.add_control(thrust)

# States
h = ca.MX.sym('h', 1)
v = ca.MX.sym('v', 1)
m = ca.MX.sym('m', 1)
goddard.add_state(h, v)
goddard.add_state(v, (thrust - sigma * v**2 * ca.exp(-h / h_ref))/m - g)
goddard.add_state(m, -thrust / c)

# Boundary Values
h_0 = ca.MX.sym('h_0', 1)
v_0 = ca.MX.sym('v_0', 1)
m_0 = ca.MX.sym('m_0', 1)
m_f = ca.MX.sym('m_f', 1)
eps_thrust = ca.MX.sym('eps_thrust')

goddard.add_constant(h_0, 0)
goddard.add_constant(v_0, 0)
goddard.add_constant(m_0, 3)
goddard.add_constant(m_f, 2.95)
goddard.add_constant(eps_thrust, 0.01)

goddard.set_cost(0, 0, -h)

goddard.add_constraint('initial', t)
goddard.add_constraint('initial', h - h_0)
goddard.add_constraint('initial', v - v_0)
goddard.add_constraint('initial', m - m_0)

goddard.add_constraint('terminal', m - m_f)

goddard.add_inequality_constraint(
        'control', thrust, lower_limit=0, upper_limit=max_thrust,
        regularizer=giuseppe.regularization.AdiffControlConstraintHandler(eps_thrust * h_ref, method='sin'))

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_ocp = giuseppe.problems.AdiffOCP(goddard)
    adiff_dual = giuseppe.problems.AdiffDual(adiff_ocp)
    adiff_dualocp = giuseppe.problems.AdiffDualOCP(adiff_ocp, adiff_dual)
    comp_dualocp = giuseppe.problems.CompDualOCP(adiff_dualocp)
    num_solver = giuseppe.numeric_solvers.AdiffScipySolveBVP(adiff_dualocp, verbose=False)
    # num_solver = giuseppe.numeric_solvers.ScipySolveBVP(comp_dualocp, verbose=False)

# guess = giuseppe.guess_generators.auto_propagate_guess(adiff_dualocp, control=80/180*3.14159)
guess = giuseppe.guess_generators.auto_linear_guess(adiff_dualocp)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = giuseppe.io.SolutionSet(adiff_dualocp, seed_sol)

cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(10, {'m_f': 1})
cont.add_logarithmic_series(20, {'eps_thrust': 1e-4}, bisection=True)
sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')
