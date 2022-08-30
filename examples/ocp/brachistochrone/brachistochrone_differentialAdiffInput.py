import numpy as np
import casadi as ca

from giuseppe.continuation import ContinuationHandler
from giuseppe.guess_generators import generate_constant_guess
from giuseppe.io import AdiffInputOCP, SolutionSet
from giuseppe.numeric_solvers.bvp import AdiffScipySolveBVP
from giuseppe.problems.dual import AdiffDual, AdiffDualOCP
from giuseppe.problems.ocp import AdiffOCP
from giuseppe.utils import Timer

input_ocp = AdiffInputOCP()

# Independent Variable
t = ca.SX.sym('t', 1)
input_ocp.set_independent(t)

# Control
tha = ca.SX.sym('θ', 1)
input_ocp.add_control(tha)

# Known Constant Parameters
g = ca.SX.sym('g', 1)
input_ocp.add_constant(g, 32.2)

# States
x = ca.SX.sym('x', 1)
y = ca.SX.sym('y', 1)
v = ca.SX.sym('v', 1)

input_ocp.add_state(x, v*ca.cos(tha))
input_ocp.add_state(y, v*ca.sin(tha))
input_ocp.add_state(v, -g*ca.sin(tha))

# Boundary Conditions
x_0 = ca.SX.sym('x_0', 1)
y_0 = ca.SX.sym('y_0', 1)
v_0 = ca.SX.sym('v_0', 1)
input_ocp.add_constant(x_0, 0)
input_ocp.add_constant(y_0, 0)
input_ocp.add_constant(v_0, 1)

x_f = ca.SX.sym('x_f', 1)
y_f = ca.SX.sym('y_f', 1)
input_ocp.add_constant(x_f, 1)
input_ocp.add_constant(y_f, -1)

input_ocp.set_cost(0, 0, t)

input_ocp.add_constraint('initial', t)
input_ocp.add_constraint('initial', x - x_0)
input_ocp.add_constraint('initial', y - y_0)
input_ocp.add_constraint('initial', v - v_0)

input_ocp.add_constraint('terminal', x - x_f)
input_ocp.add_constraint('terminal', y - y_f)

with Timer(prefix='Compilation Time:'):
    adiff_ocp = AdiffOCP(input_ocp)
    adiff_dual = AdiffDual(adiff_ocp)
    adiff_bvp = AdiffDualOCP(adiff_ocp, adiff_dual, control_method='differential')
    num_solver = AdiffScipySolveBVP(adiff_bvp)

guess = generate_constant_guess(
        adiff_bvp, t_span=0.25, x=np.array([0., 0., 1.]), lam=-0.1, u=-0.5, nu_0=-0.1, nu_f=-0.1)

seed_sol = num_solver.solve(guess.k, guess)
sol_set = SolutionSet(adiff_bvp, seed_sol)
cont = ContinuationHandler(sol_set)
cont.add_linear_series(5, {'x_f': 30, 'y_f': -30}, bisection=True)

sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')
