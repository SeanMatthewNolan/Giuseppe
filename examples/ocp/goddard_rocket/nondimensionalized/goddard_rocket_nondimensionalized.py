import numpy as np
import casadi as ca

import giuseppe as giu

# ----------------------------------------------------------------------------------------------------------------------
# The Goddard Rocket Problem formulation is taken from:
# https://www.mcs.anl.gov/~more/cops/cops3.pdf
# ----------------------------------------------------------------------------------------------------------------------

g0 = 1.0  # Gravity at surface [-]
h0 = 1.0  # Initial height
v0 = 0.0  # Initial velocity
m0 = 1.0  # Initial mass
Tc = 3.5  # Use for thrust
Hc = 500  # Use for drag
Vc = 620  # Use for drag
Mc = 0.6  # Fraction of initial mass left at end
c = 0.5 * np.sqrt(g0 * h0)  # Thrust-to-fuel mass
mf = Mc * m0  # Final mass
Dc = 0.5 * Vc * m0 / g0  # Drag scaling
T_max = Tc * g0 * m0  # Maximum thrust

iocp = giu.problems.input.ADiffInputProb(dtype=ca.SX)

t = ca.SX.sym('t', 1)
T = ca.SX.sym('T', 1)
h = ca.SX.sym('h', 1)
v = ca.SX.sym('v', 1)
m = ca.SX.sym('m', 1)
eps = ca.SX.sym('eps', 1)
vf = ca.SX.sym('vf', 1)
mfSym = ca.SX.sym('mf', 1)

drag = 1 * Dc * v ** 2 * np.exp(-Hc * (h - h0) / h0)
g = g0 * (h0 / h) ** 2

iocp.set_independent(t)
iocp.add_control(T)
iocp.add_state(h, v)
iocp.add_state(v, (T - drag) / m - g)
iocp.add_state(m, -T/c)
iocp.add_constant(eps, 0.01 * Hc)
iocp.add_constant(vf, 0)
iocp.add_constant(mfSym, mf)

iocp.add_constraint('initial', t)
iocp.add_constraint('initial', h - h0)
iocp.add_constraint('initial', v - v0)
iocp.add_constraint('initial', m - m0)
iocp.add_constraint('terminal', v - vf)
iocp.add_constraint('terminal', m - mfSym)

iocp.add_inequality_constraint(
    'control', T, lower_limit=0, upper_limit=T_max,
    regularizer=giu.problems.automatic_differentiation.regularization.ADiffControlConstraintHandler(eps, method='sin')
)

iocp.set_cost(0, 0, -h)

adiff_dual = giu.problems.automatic_differentiation.ADiffDual(iocp)
num_solver = giu.numeric_solvers.SciPySolver(adiff_dual, verbose=False)
guess = giu.guess_generation.auto_propagate_guess(adiff_dual, t_span=.1, control=0.)

seed_sol = num_solver.solve(guess)

cont = giu.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(10, {'mf': mf, 'vf': 0})
cont.add_logarithmic_series(10, {'eps': 1e-6})
sol_set = cont.run_continuation()

sol_set.save('sol_set.data')

sol = sol_set.solutions[-1]
time = sol.t
h = sol.x[0, :]
v = sol.x[1, :]
m = sol.x[2, :]

pseudo2control = adiff_dual.ca_pseudo2control.map(len(sol.t))
T = np.asarray(pseudo2control(sol.u[0, :], sol.k))

# Calculate necessary variables
Dc = 0.5 * 620 * 1.0 / 1.0
drag = 1 * Dc * v ** 2 * np.exp(-500 * (h - 1.0) / 1.0)
g = 1.0 * (1.0 / h) ** 2
