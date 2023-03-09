import casadi as ca
import numpy as np

import giuseppe

ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb(dtype=ca.SX)

# Independent Variables
t = ca.SX.sym('t', 1)
ocp.set_independent(t)

# Constants
rho_0 = ca.SX.sym('ρ_0', 1)
h_ref = ca.SX.sym('h_ref', 1)
re = ca.SX.sym('re', 1)
m = ca.SX.sym('m', 1)
mu = ca.SX.sym('µ', 1)

ocp.add_constant(rho_0, 0.002378)
ocp.add_constant(h_ref, 23_800)
ocp.add_constant(re, 20_902_900)
ocp.add_constant(m, 203_000 / 32.174)
ocp.add_constant(mu, 0.14076539e17)

a_0 = ca.SX.sym('a_0', 1)
a_1 = ca.SX.sym('a_1', 1)
b_0 = ca.SX.sym('b_0', 1)
b_1 = ca.SX.sym('b_1', 1)
b_2 = ca.SX.sym('b_2', 1)
s_ref = ca.SX.sym('s_ref', 1)

ocp.add_constant(a_0, -0.20704)
ocp.add_constant(a_1, 0.029244)
ocp.add_constant(b_0, 0.07854)
ocp.add_constant(b_1, -0.61592e-2)
ocp.add_constant(b_2, 0.621408e-3)
ocp.add_constant(s_ref, 2690)

ξ = ca.SX.sym('ξ', 1)
ocp.add_constant(ξ, 0)

eps_alpha = ca.SX.sym('ε_α', 1)
alpha_min = ca.SX.sym('α_min', 1)
alpha_max = ca.SX.sym('α_max', 1)

ocp.add_constant(eps_alpha, 1e-7)
ocp.add_constant(alpha_min, -90 / 180 * 3.1419)
ocp.add_constant(alpha_max, 90 / 180 * 3.1419)

eps_beta = ca.SX.sym('ε_β', 1)
beta_min = ca.SX.sym('β_min', 1)
beta_max = ca.SX.sym('β_max', 1)

ocp.add_constant(eps_beta, 1e-7)
ocp.add_constant(beta_min, -90 / 180 * 3.1419)
ocp.add_constant(beta_max, 90 / 180 * 3.1419)

# Add Controls
alpha = ca.SX.sym('α', 1)
beta = ca.SX.sym('β', 1)

ocp.add_control(alpha)
ocp.add_control(beta)

# State Variables
h = ca.SX.sym('h', 1)
φ = ca.SX.sym('φ', 1)
θ = ca.SX.sym('θ', 1)
v = ca.SX.sym('v', 1)
γ = ca.SX.sym('γ', 1)
ψ = ca.SX.sym('ψ', 1)

# Expressions
r = re + h
g = mu / r**2
rho = rho_0 * ca.exp(-h / h_ref)
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
ocp.add_inequality_constraint(
        'path', beta, lower_limit=beta_min, upper_limit=beta_max,
        regularizer=giuseppe.problems.automatic_differentiation.regularization.ADiffPenaltyConstraintHandler(
                eps_beta, method='sec'))

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_dual = giuseppe.problems.automatic_differentiation.ADiffDual(ocp)

x0, u0, p, k = np.array([260_000, 100., 100., 2500, -1/180*3.14, 0]), np.array([0.1, 0]), np.array([]),\
    adiff_dual.default_values
x_dot = adiff_dual.compute_dynamics(12., x0, u0, p, k)
psi_0 = adiff_dual.compute_initial_boundary_conditions(12., x0, p, k)
psi_f = adiff_dual.compute_terminal_boundary_conditions(12., x0, p, k)
lam0, nu0, nuf = np.ones_like(x0), np.ones((7,)), np.ones((3,))
lam_dot = adiff_dual.compute_costate_dynamics(100, x0, lam0, u0, p, k)
adj_bc0 = adiff_dual.compute_initial_adjoint_boundary_conditions(100, x0, lam0, u0, p, nu0, k)
adj_bcf = adiff_dual.compute_terminal_adjoint_boundary_conditions(100, x0, lam0, u0, p, nuf, k)

solver = giuseppe.numeric_solvers.SciPySolver(adiff_dual)
guess = giuseppe.guess_generation.auto_propagate_guess(adiff_dual, control=(15/180*3.14159, 0), t_span=100)

cont = giuseppe.continuation.ContinuationHandler(solver, guess)
cont.add_linear_series(100, {'h_f': 150_000, 'v_f': 15_000})
cont.add_linear_series(50, {'h_f': 80_000, 'v_f': 2_500, 'γ_f': -5 / 180 * 3.14159})
cont.add_linear_series(90, {'ξ': np.pi / 2})
cont.add_linear_series(90, {'β_min': -70 / 180 * 3.14159, 'β_max': 70 / 180 * 3.14159})
cont.add_logarithmic_series(90, {'ε_α': 1e-10, 'ε_β': 1e-10})
sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
