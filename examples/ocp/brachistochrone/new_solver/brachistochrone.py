import os

import numpy as np

from giuseppe import SymDual, StrInputProb, auto_propagate_guess, auto_guess, SciPySolver
from giuseppe.numeric_solvers.dual.collocation import DualCollocation
from giuseppe.data_classes.remesh import remesh

os.chdir(os.path.dirname(__file__))  # Set directory to file location

ocp = StrInputProb()

ocp.set_independent('t')

ocp.add_state('x', 'v*cos(theta)')
ocp.add_state('y', 'v*sin(theta)')
ocp.add_state('v', '-g*sin(theta)')

ocp.add_control('theta')

ocp.add_constant('g', 32.2)

ocp.add_constant('x_0', 0)
ocp.add_constant('y_0', 0)
ocp.add_constant('v_0', 1)

ocp.add_constant('x_f', 1)
ocp.add_constant('y_f', -1)

ocp.set_cost('0', '1', '0')

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'x - x_0')
ocp.add_constraint('initial', 'y - y_0')
ocp.add_constraint('initial', 'v - v_0')

ocp.add_constraint('terminal', 'x - x_f')
ocp.add_constraint('terminal', 'y - y_f')

comp_dual = SymDual(ocp)
guess = auto_guess(comp_dual, u=-15 / 180 * 3.14159)

guess = SciPySolver(comp_dual).solve(guess)

remeshed = remesh(guess, np.linspace(guess.t[0], guess.t[-1], 11))

solver = DualCollocation(comp_dual)
out = solver.solve(remeshed)
