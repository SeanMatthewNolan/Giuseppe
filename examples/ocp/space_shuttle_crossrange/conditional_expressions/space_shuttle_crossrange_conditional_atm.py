import casadi as ca
import numpy as np

import giuseppe
from giuseppe.utils.examples.atmosphere1976 import Atmosphere1976

ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb()

# Independent Variables
t = ca.SX.sym('t', 1)
ocp.set_independent(t)

# Constants
rho_0 = ca.SX.sym('rho_0')
h_ref = ca.SX.sym('h_ref')
re = ca.SX.sym('re')
mu = ca.SX.sym('mu')

ocp.add_constant(rho_0, 0.002378)
ocp.add_constant(h_ref, 23_800.)
ocp.add_constant(re, 20_902_900.)
ocp.add_constant(mu, 0.14076539e17)

g0 = 0.14076539e17 / 20_902_900.**2

a_0 = -0.20704
a_1 = 0.029244
b_0 = 0.07854
b_1 = -0.61592e-2
b_2 = 0.621408e-3

# Mutable constants
m = ca.SX.sym('m', 1)
s_ref = ca.SX.sym('s_ref', 1)
ξ = ca.SX.sym('ξ', 1)

ocp.add_constant(m, 203_000 / 32.174)
ocp.add_constant(s_ref, 2690)
ocp.add_constant(ξ, 0)

eps_alpha = ca.SX.sym('ε_α', 1)
alpha_min = ca.SX.sym('α_min', 1)
alpha_max = ca.SX.sym('α_max', 1)

ocp.add_constant(eps_alpha, 1e-5)
ocp.add_constant(alpha_min, -80 / 180 * 3.1419)
ocp.add_constant(alpha_max, 80 / 180 * 3.1419)

eps_beta = ca.SX.sym('ε_β', 1)
beta_min = ca.SX.sym('β_min', 1)
beta_max = ca.SX.sym('β_max', 1)

ocp.add_constant(eps_beta, 1e-10)
ocp.add_constant(beta_min, -85 / 180 * 3.1419)
ocp.add_constant(beta_max, 85 / 180 * 3.1419)

# State Variables
h = ca.SX.sym('h', 1)
φ = ca.SX.sym('φ', 1)
θ = ca.SX.sym('θ', 1)
v = ca.SX.sym('v', 1)
γ = ca.SX.sym('γ', 1)
ψ = ca.SX.sym('ψ', 1)

# Atmosphere Func
atm = Atmosphere1976(use_metric=False, earth_radius=re, gravity=g0, boundary_thickness=1000.)
_, __, rho_conditional = atm.get_ca_atm_expr(h)

rho_exponential = rho_0 * ca.exp(-h / h_ref)

conditional = ca.SX.sym('conditional', 1)
ocp.add_constant(conditional, 0)

rho = conditional * rho_conditional + (1 - conditional) * rho_exponential

# Add Controls
alpha = ca.SX.sym('α', 1)
beta = ca.SX.sym('β', 1)

ocp.add_control(alpha)
ocp.add_control(beta)

# Expressions
r = re + h
g = mu / r**2
# rho = rho_0 * ca.exp(-h / h_ref)
dyn_pres = 1 / 2 * rho * v ** 2
alpha_hat = alpha * 180 / ca.pi
c_l = a_0 + a_1 * alpha_hat
c_d = b_0 + b_1 * alpha_hat + b_2 * alpha_hat**2
lift = c_l * s_ref * dyn_pres
drag = c_d * s_ref * dyn_pres

# Add States & EOMs
ocp.add_state(h, v * ca.sin(γ))
ocp.add_state(φ, v * ca.cos(γ) * ca.sin(ψ) / (r * ca.cos(θ)))
ocp.add_state(θ, v * ca.cos(γ) * ca.cos(ψ) / r)
ocp.add_state(v, -drag / m - g * ca.sin(γ))
ocp.add_state(γ, lift * ca.cos(beta) / (m * v) + ca.cos(γ) * (v / r - g / v))
ocp.add_state(ψ, lift * ca.sin(beta)/(m * v * ca.cos(γ)) + v * ca.cos(γ) * ca.sin(ψ) * ca.sin(θ)/(r * ca.cos(θ)))

# Cost
ocp.set_cost(0, 0, -φ * ca.cos(ξ) - θ * ca.sin(ξ))

# Boundary Values
h_0 = ca.SX.sym('h_0', 1)
φ_0 = ca.SX.sym('φ_0', 1)
θ_0 = ca.SX.sym('θ_0', 1)
v_0 = ca.SX.sym('v_0', 1)
γ_0 = ca.SX.sym('γ_0', 1)
ψ_0 = ca.SX.sym('ψ_0', 1)

ocp.add_constant(h_0, 260_000)
ocp.add_constant(φ_0, 0)
ocp.add_constant(θ_0, 0)
ocp.add_constant(v_0, 25_600)
ocp.add_constant(γ_0, -1 / 180 * np.pi)
ocp.add_constant(ψ_0, np.pi / 2)

h_f = ca.SX.sym('h_f', 1)
v_f = ca.SX.sym('v_f', 1)
γ_f = ca.SX.sym('γ_f', 1)

ocp.add_constant(h_f, 80_000)
ocp.add_constant(v_f, 2_500)
ocp.add_constant(γ_f, -5 / 180 * np.pi)

ocp.add_constraint('initial', t)
ocp.add_constraint('initial', h - h_0)
ocp.add_constraint('initial', φ - φ_0)
ocp.add_constraint('initial', θ - θ_0)
ocp.add_constraint('initial', v - v_0)
ocp.add_constraint('initial', γ - γ_0)
ocp.add_constraint('initial', ψ - ψ_0)

ocp.add_constraint('terminal', h - h_f)
ocp.add_constraint('terminal', v - v_f)
ocp.add_constraint('terminal', γ - γ_f)

ocp.add_inequality_constraint(
        'path', alpha, lower_limit=alpha_min, upper_limit=alpha_max,
        regularizer=giuseppe.problems.automatic_differentiation.regularization.ADiffPenaltyConstraintHandler(
                eps_alpha, method='sec'))
# ocp.add_inequality_constraint('path', beta, lower_limit=beta_min, upper_limit=beta_max,
#                               regularizer=giuseppe.regularization.AdiffPenaltyConstraintHandler(
#                                   eps_alpha, method='sec'))

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_dual = giuseppe.problems.automatic_differentiation.ADiffDual(ocp)
    num_solver = giuseppe.numeric_solvers.SciPySolver(
        adiff_dual, verbose=False, max_nodes=100, node_buffer=10
    )

if __name__ == '__main__':
    guess = giuseppe.guess_generation.auto_propagate_guess(adiff_dual, control=(20/180*3.14159, 0), t_span=100)
    # guess.k[-3:] = guess.x[(0, 3, 4), -1]  # match h_f, v_f, gam_f
    seed_sol = num_solver.solve(guess)

    cont1 = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    cont1.add_linear_series(100, {'h_f': 200_000, 'v_f': 20_000})
    cont1.add_linear_series(50, {'h_f': 80_000, 'v_f': 2_500, 'γ_f': -5 * np.pi / 180})
    cont1.add_linear_series(90, {'ξ': 0.5 * np.pi}, bisection=True)
    sol_set1 = cont1.run_continuation()

    sol_set1.save('sol_set.data')

    cont2 = giuseppe.continuation.ContinuationHandler(num_solver, sol_set1.solutions[-1])
    cont2.add_linear_series(10, {'conditional': 1}, bisection=True)
    sol_set2 = cont2.run_continuation()

    sol_set2.save('sol_set_conditional.data')
