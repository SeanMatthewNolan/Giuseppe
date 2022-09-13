import casadi as ca
import numpy as np

import giuseppe

zermelo = giuseppe.io.AdiffInputOCP()

# Independent Var
t = ca.MX.sym('t')
zermelo.set_independent(t)

# Control
theta = ca.MX.sym('theta')
zermelo.add_control(theta)

# States
x = ca.MX.sym('x')
y = ca.MX.sym('y')

# Constants
v = ca.MX.sym('v')
c = ca.MX.sym('c')
x_0 = ca.MX.sym('x_0')
y_0 = ca.MX.sym('y_0')
x_f = ca.MX.sym('x_f')
y_f = ca.MX.sym('y_f')

zermelo.add_constant(v, 1)
zermelo.add_constant(c, 0)

zermelo.add_constant(x_0, 3.5)
zermelo.add_constant(y_0, -1.8)

zermelo.add_constant(x_f, 0.)
zermelo.add_constant(y_f, 0.)

# Lookup Table (current = c*y)
interp_method = 'linear'  # either 'bspline' or 'linear'

y_breakpoints = np.array((-2, -1, 0, 1, 2))
c_breakpoints = np.array((-2, -1, 0, 1, 2))
current_data = np.array([[y_breakpoint * c_breakpoint
                         for y_breakpoint in y_breakpoints]
                         for c_breakpoint in c_breakpoints])
current_data_flat = current_data.ravel(order='F')
current_table = ca.interpolant('current_table', interp_method, (y_breakpoints, c_breakpoints), current_data_flat)
current = current_table(ca.vcat((y, c)))

# EoMs
zermelo.add_state(x, v*ca.cos(theta) + current)
zermelo.add_state(y, v*ca.sin(theta))

# Cost (Min. Time)
zermelo.set_cost(0, 0, t)

# Boundary Conditions
zermelo.add_constraint('initial', t)
zermelo.add_constraint('initial', x - x_0)
zermelo.add_constraint('initial', y - y_0)

zermelo.add_constraint('terminal', x - x_f)
zermelo.add_constraint('terminal', y - y_f)

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_ocp = giuseppe.problems.AdiffOCP(zermelo)
    adiff_dual = giuseppe.problems.AdiffDual(adiff_ocp)
    adiff_dual_ocp = giuseppe.problems.AdiffDualOCP(adiff_ocp, adiff_dual, control_method='differential')
    num_solver = giuseppe.numeric_solvers.AdiffScipySolveBVP(adiff_dual_ocp, verbose=False)

# guess = giuseppe.guess_generators.generate_constant_guess(comp_dual_ocp)
guess = giuseppe.guess_generators.auto_propagate_guess(adiff_dual_ocp, control=0, t_span=1)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = giuseppe.io.SolutionSet(adiff_dual_ocp, seed_sol)

cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(10, {'x_f': 0, 'y_f': 0})
cont.add_linear_series(6, {'c': -1})
cont.add_linear_series(6, {'c': 1})

cont.run_continuation(num_solver)

sol_set.save('current_variation_sol_set.data')
