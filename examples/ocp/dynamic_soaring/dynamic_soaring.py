import numpy as np
import giuseppe
import pickle

soar = giuseppe.problems.input.StrInputProb()

soar.set_independent('t')
soar.set_cost('0', '0', 'wind_gradient')
soar.add_parameter('wind_gradient')

soar.add_state('x', 'V * cos(gam) * sin(psi) + Wx')
soar.add_state('y', 'V * cos(gam) * cos(psi)')
soar.add_state('z', 'V * sin(gam)')
soar.add_state('V', '-drag/mass - g * sin(gam) - dWxdt * cos(gam) * sin(psi)')
soar.add_state('gam', 'lift * cos(bank) / (mass * V) - g * cos(gam) / V + dWxdt * sin(gam) * sin(psi) / V')
soar.add_state('psi', 'lift * sin(bank) / (mass * V * cos(gam)) * dWxdt * cos(psi) / (V * cos(gam))')

soar.add_control('CL')
soar.add_control('bank')

soar.add_expression('qdyn', '0.5 * rho * V ** 2')
soar.add_expression('drag', 'qdyn * s_ref * (CD0 + CD1 * CL ** 2)')
soar.add_expression('lift', 'qdyn * s_ref * CL')
soar.add_expression('Wx', 'wind_gradient * z')
soar.add_expression('dWxdt', 'wind_gradient * V * sin(gam)')

soar.add_constant('g', 32.2)
soar.add_constant('rho', 0.002378)
soar.add_constant('s_ref', 45.09703)
soar.add_constant('mass', 5.6)
soar.add_constant('CD0', 0.00873)
soar.add_constant('CD1', 0.045)

# Boundary Conditions
soar.add_parameter('V0')
soar.add_parameter('gam0')
soar.add_parameter('psi0')

soar.add_constraint('initial', 't')
soar.add_constraint('initial', 'x')
soar.add_constraint('initial', 'y')
soar.add_constraint('initial', 'z')
soar.add_constraint('initial', 'V - V0')
soar.add_constraint('initial', 'gam - gam0')
soar.add_constraint('initial', 'psi - psi0')

soar.add_constant('xf', 0.0)
soar.add_constant('yf', 0.0)
soar.add_constant('zf', 0.0)
soar.add_constant('dV', 0.0)
soar.add_constant('dgam', 0.0)
soar.add_constant('dpsi', -2 * np.pi)

soar.add_constraint('terminal', 'x - xf')
soar.add_constraint('terminal', 'y - yf')
soar.add_constraint('terminal', 'z - zf')
soar.add_constraint('terminal', 'V - (V0 + dV)')
soar.add_constraint('terminal', 'gam - (gam0 + dgam)')
soar.add_constraint('terminal', 'psi - (psi0 + dpsi)')

# For uniqueness of solution, apply exterior penalty to psi0
soar.add_inequality_constraint(
    'path', 'psi0', lower_limit='psi0_min', upper_limit='psi0_max',
    regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler('eps_psi0', method='utm')
)
soar.add_constant('psi0_min', -np.pi/2)
soar.add_constant('psi0_max', 2*np.pi + np.pi/2)
soar.add_constant('eps_psi0', 1e0)

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_soar = giuseppe.problems.symbolic.SymDual(soar, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_soar, verbose=True, max_nodes=100, node_buffer=10)

wind_guess = 0.1
V0_guess = 100.0
gam0_guess = 0.0
psi0_guess = 0.0

p_guess = np.array((wind_guess, V0_guess, gam0_guess, psi0_guess))
x0_guess = np.array((0., 0., 0., V0_guess, gam0_guess, psi0_guess))
guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_soar, control=np.array((0.3, 15 * np.pi / 180)), t_span=1.0, verbose=True, p=p_guess, initial_states=x0_guess,
    match_parameters=False
)

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

d2r = np.pi/180

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)

psi1 = -1 * d2r
gam1 = 1 * d2r
cont.add_linear_series(100, {'dgam': gam1}, bisection=True)
cont.add_linear_series(100, {'xf': seed_sol.p[1] * np.sin(psi1), 'yf': seed_sol.p[1] * np.cos(psi1), 'dpsi': psi1}, bisection=True)

sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
