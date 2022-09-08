import numpy as np
import casadi as ca

import giuseppe

ocp = giuseppe.io.AdiffInputOCP()

# Independent Variable
t = ca.SX.sym('t', 1)
ocp.set_independent(t)

# Control
alpha = ca.SX.sym('alpha', 1)
ocp.add_control(alpha)

# Known Constant Parameters
Isp = ca.SX.sym('Isp', 1)
g0 = ca.SX.sym('g0', 1)
S = ca.SX.sym('S', 1)
mu = ca.SX.sym('mu', 1)
Re = ca.SX.sym('Re', 1)

ocp.add_constant(Isp, 16000.0)  # s
ocp.add_constant(g0, 32.174)  # ft/s^2
ocp.add_constant(S, 530.0)  # ft^2
ocp.add_constant(mu, 1.4076539e16)  # ft^3/s^2
ocp.add_constant(Re, 20902900.0)  # ft

# States
h = ca.MX.sym('h', 1)
v = ca.MX.sym('v', 1)
gam = ca.SX.sym('gam', 1)
w = ca.SX.sym('w', 1)

# TODO add atmosphere model
M = v / 1125.33  # assume a = 343 m/s = 1125.33 ft/s

# Look-Up Tables
interp_method = 'bspline'  # either 'bspline' or 'linear'

M_grid_thrust = np.array((0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8))
h_grid_thrust = np.array((0, 5, 10, 15, 20, 25, 30, 40, 50, 70)) * 1e3
data_thrust = np.array(((24.2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, 0, 0, 0),
                        (28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, 0),
                        (30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, 0),
                        (34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1),
                        (37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4),
                        (36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7),
                        (0, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0, 2.2),
                        (0, 0, 0, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9),
                        (0, 0, 0, 0, 0, 34.6, 31.1, 21.7, 13.3, 3.1)))

data_flat_thrust = data_thrust.ravel(order='F')
thrust_table = ca.interpolant('thrust_table', interp_method, (M_grid_thrust, h_grid_thrust), data_flat_thrust)

thrust = thrust_table(ca.vcat((M, h)))

M_grid_aero = np.array((0, 0.4, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8))
data_CLalpha = np.array((3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44))
data_CD0 = np.array((0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035))
data_eta = np.array((0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93))

CLalpha_table = ca.interpolant('CLalpha_table', interp_method, (M_grid_aero,), data_CLalpha)
CD0_table = ca.interpolant('CLalpha_table', interp_method, (M_grid_aero,), data_CLalpha)
eta_table = ca.interpolant('CLalpha_table', interp_method, (M_grid_aero,), data_CLalpha)

CLalpha = CLalpha_table(M)
CD0 = CD0_table(M)
eta = eta_table(M)

# Inequality Constraints
h_min = ca.SX.sym('h_min', 1)
h_max = ca.SX.sym('h_max', 1)
eps_h = ca.SX.sym('eps_h', 1)

gam_min = ca.SX.sym('gam_min', 1)
gam_max = ca.SX.sym('gam_max', 1)
eps_gam = ca.SX.sym('eps_gam', 1)

v_min = ca.SX.sym('v_min', 1)
v_max = ca.SX.sym('v_max', 1)
eps_v = ca.SX.sym('eps_v', 1)

w_min = ca.SX.sym('w_min', 1)
w_max = ca.SX.sym('w_max', 1)
eps_w = ca.SX.sym('eps_w', 1)

alpha_min = ca.SX.sym('alpha_min', 1)
alpha_max = ca.SX.sym('alpha_max', 1)
eps_alpha = ca.SX.sym('eps_alpha', 1)

ocp.add_inequality_constraint('path', h,
                              lower_limit=h_min, upper_limit=h_max,
                              regularizer=giuseppe.regularization.AdiffPenaltyConstraintHandler(regulator=eps_h))
ocp.add_inequality_constraint('path', gam,
                              lower_limit=gam_min, upper_limit=gam_max,
                              regularizer=giuseppe.regularization.AdiffPenaltyConstraintHandler(regulator=eps_gam))
ocp.add_inequality_constraint('path', v,
                              lower_limit=v_min, upper_limit=v_max,
                              regularizer=giuseppe.regularization.AdiffPenaltyConstraintHandler(regulator=eps_v))
ocp.add_inequality_constraint('path', w,
                              lower_limit=w_min, upper_limit=w_max,
                              regularizer=giuseppe.regularization.AdiffPenaltyConstraintHandler(regulator=eps_w))
ocp.add_inequality_constraint('path', alpha,
                              lower_limit=alpha_min, upper_limit=alpha_max,
                              regularizer=giuseppe.regularization.AdiffControlConstraintHandler(regulator=eps_alpha))

# Initial Boundary Conditions
h_0 = ca.SX.sym('h_0', 1)
v_0 = ca.SX.sym('v_0', 1)
gam_0 = ca.SX.sym('gam_0', 1)
w_0 = ca.SX.sym('w_0', 1)

ocp.add_constant(h_0, 0.0)  # ft
ocp.add_constant(v_0, 424.26)  # ft/s
ocp.add_constant(gam_0, 0.0)  # rad
ocp.add_constant(w_0, 42000.0)  # lb

ocp.add_constraint(location='initial', expr=h - h_0)
ocp.add_constraint(location='initial', expr=v - v_0)
ocp.add_constraint(location='initial', expr=gam - gam_0)
ocp.add_constraint(location='initial', expr=w - w_0)

# Terminal Boundary Conditions
h_f = ca.SX.sym('h_f', 1)
v_f = ca.SX.sym('v_f', 1)
gam_f = ca.SX.sym('gam_f', 1)

ocp.add_constant(h_f, 65600.0)  # ft
ocp.add_constant(v_f, 968.148)  # ft/s
ocp.add_constant(gam_f, 0.0)  # rad

ocp.add_constraint(location='terminal', expr=h - h_f)
ocp.add_constraint(location='terminal', expr=v - v_f)
ocp.add_constraint(location='terminal', expr=gam - gam_f)

# Compilation

# Continuation and Solving

