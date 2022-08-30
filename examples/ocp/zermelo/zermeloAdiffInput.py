import casadi as ca

import giuseppe

zermelo = giuseppe.io.AdiffInputOCP()

# Set Independent var
t = ca.SX.sym('t', 1)
zermelo.set_independent(t)

# Controls
theta = ca.SX.sym('tha', 1)
zermelo.add_control(theta)

# Known Constant Parameters
v = ca.SX.sym('v', 1)
c = ca.SX.sym('c', 1)
zermelo.add_constant(v, 1)
zermelo.add_constant(c, -1)

# States
x = ca.SX.sym('x', 1)
y = ca.SX.sym('y', 1)

current = c * y

zermelo.add_state(x, v*ca.cos(theta) + current)
zermelo.add_state(y, v*ca.sin(theta))

# Boundary Conditions
x_0 = ca.SX.sym('x_0', 1)
y_0 = ca.SX.sym('y_0', 1)
zermelo.add_constant(x_0, 3.5)
zermelo.add_constant(y_0, -1.8)

x_f = ca.SX.sym('x_f', 1)
y_f = ca.SX.sym('y_f', 1)
zermelo.add_constant(x_f, 0.)
zermelo.add_constant(y_f, 0.)

zermelo.set_cost(0, 0, t)

zermelo.add_constraint('initial', t)
zermelo.add_constraint('initial', x - x_0)
zermelo.add_constraint('initial', y - y_0)

zermelo.add_constraint('terminal', x - x_f)
zermelo.add_constraint('terminal', y - y_f)

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_ocp = giuseppe.problems.AdiffOCP(zermelo)
    adiff_dual = giuseppe.problems.AdiffDual(adiff_ocp)
    adiff_bvp = giuseppe.problems.AdiffDualOCP(adiff_ocp, adiff_dual)
    num_solver = giuseppe.numeric_solvers.AdiffScipySolveBVP(adiff_bvp)

guess = giuseppe.guess_generators.generate_constant_guess(adiff_bvp)
sol = num_solver.solve(guess.k, guess)
sol.save('zermelo.bson')
