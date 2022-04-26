import pickle
import numpy as np
from sympy.functions.special.delta_functions import Heaviside

from giuseppe.continuation import ContinuationHandler, SolutionSet
from giuseppe.guess_generators import auto_propagate_guess
from giuseppe.io import InputOCP
from giuseppe.numeric_solvers.bvp import ScipySolveBVP
from giuseppe.problems.dual import SymDual, SymDualOCP, CompDualOCP
from giuseppe.problems.ocp import SymOCP
from giuseppe.utils import Timer

ocp = InputOCP()

ocp.set_independent('t')

rho = 'rho0*exp(-h/Hscale)'
cd = '(cd0 + cd1 * alpha + cd2 * alpha**2)'
cl = '(cl0 + cl1 * alpha)'
q = '0.5 * ' + rho + '* v**2'
L = q + ' * Sref * ' + cl
D = q + ' * Sref * ' + cd

ocp.add_state('h', 'v*sin(gamma)')
ocp.add_state('theta', 'v*cos(gamma)/(re + h)')
ocp.add_state('v', '-' + D + '/m - mu*sin(gamma)/(re + h)**2')
ocp.add_state('gamma', L + '/(m*v) + (v/(re + h) - mu/(v*(re + h)**2))*cos(gamma)')
# ocp.add_state('constraint_violation', g1 + ' * ' + g1 + '* Heaviside(' + g1 + ')')

ocp.add_control('alpha')

re = 6.371e6

ocp.add_constant('rho0', 1.2)
ocp.add_constant('Hscale', 7.5e3)
ocp.add_constant('cd0', 0.26943)
ocp.add_constant('cd1', -0.4113)
ocp.add_constant('cd2', 18.231)
ocp.add_constant('cl0', 0.1758)
ocp.add_constant('cl1', 10.305)
ocp.add_constant('Sref', 0.35)
ocp.add_constant('re', re)
ocp.add_constant('m', 1e3)
ocp.add_constant('mu', 3.986e14)
ocp.add_constant('pi', np.pi)

h_0 = 50e3
theta_0 = 0
v_0 = 3e3
gamma_0 = -10 * np.pi/180
h_f = 0
theta_f = 600e3 / re

h_0_guess = 1e3
v_0_guess = 300
theta_f_guess = 1.5e3 / re

ocp.add_constant('h_0', h_0_guess)
ocp.add_constant('theta_0', theta_0)
ocp.add_constant('v_0', v_0_guess)
ocp.add_constant('gamma_0', gamma_0)

ocp.add_constant('h_f', h_f)
ocp.add_constant('theta_f', theta_f_guess)

# Path Constraints
alpha_min = -40 * np.pi / 180
alpha_max = -alpha_min
h_min = -10
h_max = 70e3
c4_min = -10 * re
c4_max = 10 * re
theta_cr = theta_f + 100e3 / re
ocp.add_constant('alpha_min', alpha_min)
ocp.add_constant('alpha_max', alpha_max)
ocp.add_constant('h_min', h_min)
ocp.add_constant('h_max', h_max)
ocp.add_constant('C4_min', c4_min)
ocp.add_constant('C4_max', c4_max)
ocp.add_constant('theta_cr', theta_cr)
ocp.add_constant('eps_h', 1e-5)
ocp.add_constant('eps_alpha', 1.0)
ocp.add_constant('eps_sensor', 1.0)
g_h = '(eps_h / cos(pi/2 * (2 * h - h_max - h_min) / (h_max - h_min)))'
g_alpha = '(eps_alpha / cos(pi/2 * (2 * alpha - alpha_max - alpha_min) / (alpha_max - alpha_min)))'

C4 = '(re * sin(theta) - (re + h) * sin(theta_cr)' + \
     '+ 1/tan(theta_cr) * (re * cos(theta)) - (re + h) * cos(theta_cr))'
g_sensor = '(eps_sensor / cos(pi/2 * (2 * ' + C4 + ' - C4_max - C4_min) / (C4_max - C4_min)))'

# ocp.set_cost('0', '0', '-v')
ocp.set_cost('0', '(' + g_h + ' + ' + g_alpha + ' + ' + g_sensor + ')', '-v')

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'h - h_0')
ocp.add_constraint('initial', 'theta - theta_0')
ocp.add_constraint('initial', 'v - v_0')
ocp.add_constraint('initial', 'gamma - gamma_0')

ocp.add_constraint('terminal', 'h - h_f')
ocp.add_constraint('terminal', 'theta - theta_f')

# ocp.add_inequality_constraint('path', 'alpha', lower_limit='alpha_min', upper_limit='alpha_max')

with Timer(prefix='Compilation Time:'):
    sym_ocp = SymOCP(ocp)
    sym_dual = SymDual(sym_ocp)
    sym_bvp = SymDualOCP(sym_ocp, sym_dual, control_method='differential')
    comp_dual_ocp = CompDualOCP(sym_bvp, use_jit_compile=True)
    num_solver = ScipySolveBVP(comp_dual_ocp)

guess = auto_propagate_guess(comp_dual_ocp, control=0, t_span=10)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = SolutionSet(sym_bvp, seed_sol)
cont = ContinuationHandler(sol_set)
cont.add_linear_series(10, {'h_f': h_f, 'theta_f': theta_f_guess}, bisection=True)
cont.add_linear_series(10, {'v_0': 302, 'h_0': 1.01e3, 'theta_f': 1.6e3 / re}, bisection=True)
cont.add_linear_series(10, {'v_0': v_0, 'h_0': 10e3, 'theta_f': 15e3 / re}, bisection=True)
cont.add_linear_series(10, {'gamma_0': 0 * np.pi/180, 'h_0': 11e3, 'theta_f': 20e3 / re}, bisection=True)
cont.add_linear_series(10, {'h_0': 20e3, 'theta_f': 40e3 / re}, bisection=True)
cont.add_linear_series(10, {'h_0': 40e3, 'theta_f': 100e3 / re}, bisection=True)
cont.add_linear_series(10, {'theta_f': theta_f}, bisection=True)
cont.add_linear_series(100, {'alpha_min': -10 * np.pi/180, 'alpha_max': 10 * np.pi/180}, bisection=True)
cont.add_logarithmic_series(20, {'eps_alpha': 1e-5}, bisection=True)

with Timer(prefix='Continuation Time:'):
    for series in cont.continuation_series:
        for k, last_sol in series:
            sol_i = num_solver.solve(k, last_sol)
            sol_set.append(sol_i)

with open('sol_set.data', 'wb') as file:
    pickle.dump(sol_set, file)
