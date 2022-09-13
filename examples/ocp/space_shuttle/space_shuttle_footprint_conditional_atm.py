import numpy as np
import casadi as ca

import giuseppe

from atmosphere1976 import Atmosphere1976

giuseppe.utils.compilation.JIT_COMPILE = True

ocp = giuseppe.io.AdiffInputOCP()

# Independent Variables
t = ca.MX.sym('t', 1)
ocp.set_independent(t)

# Immutable Constants
rho_0 = 0.002378
h_ref = 23_800
re = 20_902_900
mu = 0.14076539e17
g0 = mu / re**2

a_0 = -0.20704
a_1 = 0.029244
b_0 = 0.07854
b_1 = -0.61592e-2
b_2 = 0.621408e-3

# Mutable constants
m = ca.MX.sym('m', 1)
s_ref = ca.MX.sym('s_ref', 1)
ξ = ca.MX.sym('ξ', 1)

ocp.add_constant(m, 203_000 / 32.174)
ocp.add_constant(s_ref, 2690)
ocp.add_constant(ξ, 0)

eps_alpha = ca.MX.sym('ε_α', 1)
alpha_min = ca.MX.sym('α_min', 1)
alpha_max = ca.MX.sym('α_max', 1)

ocp.add_constant(eps_alpha, 1e-5)
ocp.add_constant(alpha_min, -80 / 180 * 3.1419)
ocp.add_constant(alpha_max, 80 / 180 * 3.1419)

eps_beta = ca.MX.sym('ε_β', 1)
beta_min = ca.MX.sym('β_min', 1)
beta_max = ca.MX.sym('β_max', 1)

ocp.add_constant(eps_beta, 1e-10)
ocp.add_constant(beta_min, -85 / 180 * 3.1419)
ocp.add_constant(beta_max, 85 / 180 * 3.1419)

# Unit Conversions
slug2kg = 14.5939
kg2slug = 1/slug2kg
m2ft = 3.28084
ft2m = 1/m2ft
K2R = 1.8
R2K = 1 / K2R
Pa2psf = 0.020885
N2lb = 0.2248090795

# State Variables
h = ca.MX.sym('h', 1)
φ = ca.MX.sym('φ', 1)
θ = ca.MX.sym('θ', 1)
v = ca.MX.sym('v', 1)
γ = ca.MX.sym('γ', 1)
ψ = ca.MX.sym('ψ', 1)

# Atmosphere Func
atm = Atmosphere1976(use_metric=False, gravity=g0, earth_radius=re)


# TODO refactor callback into backend
class AtmosphereFunction(ca.Callback):
    def __init__(self):
        ca.Callback.__init__(self)
        self.construct('atmosphere_function', {"enable_fd": True})

    @staticmethod
    def get_n_in(): return 1

    @staticmethod
    def get_n_out(): return 3

    @staticmethod
    def init(): return

    @staticmethod
    def eval(altitude):
        return list(atm.atm_func(float(altitude[0])))


ca_atm_func = AtmosphereFunction()
rho = ca_atm_func(h)[2]  # AtmosphereFunction outputs temperature, pressure, density

# Add Controls
alpha = ca.MX.sym('α', 1)
beta = ca.MX.sym('β', 1)

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
h_0 = ca.MX.sym('h_0', 1)
φ_0 = ca.MX.sym('φ_0', 1)
θ_0 = ca.MX.sym('θ_0', 1)
v_0 = ca.MX.sym('v_0', 1)
γ_0 = ca.MX.sym('γ_0', 1)
ψ_0 = ca.MX.sym('ψ_0', 1)

ocp.add_constant(h_0, 260_000)
ocp.add_constant(φ_0, 0)
ocp.add_constant(θ_0, 0)
ocp.add_constant(v_0, 25_600)
ocp.add_constant(γ_0, -1 / 180 * np.pi)
ocp.add_constant(ψ_0, np.pi / 2)

h_f = ca.MX.sym('h_f', 1)
v_f = ca.MX.sym('v_f', 1)
γ_f = ca.MX.sym('γ_f', 1)

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

ocp.add_inequality_constraint('path', alpha, lower_limit=alpha_min, upper_limit=alpha_max,
                              regularizer=giuseppe.regularization.AdiffPenaltyConstraintHandler(
                                  eps_alpha, method='sec'))
# ocp.add_inequality_constraint('path', beta, lower_limit=beta_min, upper_limit=beta_max,
#                               regularizer=giuseppe.regularization.AdiffPenaltyConstraintHandler(
#                                   eps_alpha, method='sec'))

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_ocp = giuseppe.problems.ocp.AdiffOCP(ocp)
    adiff_dual = giuseppe.problems.AdiffDual(adiff_ocp)
    adiff_bvp = giuseppe.problems.AdiffDualOCP(adiff_ocp, adiff_dual, control_method='differential')
    num_solver = giuseppe.numeric_solvers.AdiffScipySolveBVP(adiff_bvp, bc_tol=1e-8)

guess = giuseppe.guess_generators.auto_propagate_guess(adiff_bvp, control=(20/180*3.14159, 0), t_span=100)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = giuseppe.io.SolutionSet(adiff_bvp, seed_sol)

cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(100, {'h_f': 200_000, 'v_f': 10_000})
cont.add_linear_series(50, {'h_f': 80_000, 'v_f': 2_500, 'γ_f': -5 / 180 * 3.14159})
cont.add_linear_series(90, {'ξ': np.pi / 2}, bisection=True)
sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')
