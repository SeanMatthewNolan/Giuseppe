import numpy as np
import giuseppe
import pickle

climb = giuseppe.problems.input.StrInputProb()

climb.set_independent('t')
climb.set_cost('0', '0', '(1 - frac_time_cost) * (mass0 - mass) + frac_time_cost * t')
climb.add_constant('frac_time_cost', 1.0)  # 1 = min. time, 0 = min. fuel

climb.add_state('h', 'V * sin(gam)')
climb.add_state('d', 'V * cos(gam)')
climb.add_state('V', '(thrust_frac * thrust_max - qdyn * s_ref * CD - mass * g * sin(gam)) / mass')
climb.add_state('gam', '(qdyn * s_ref * CL / mass - g * cos(gam)) / V')
climb.add_state('mass', '-CS * thrust_frac * thrust_max')

# Regularaized Control
climb.add_control('thrust_frac')
climb.add_control('CL')

thrust_frac_min = 0.3
thrust_frac_max = 1.0
climb.add_inequality_constraint(
    'control', 'thrust_frac', lower_limit='thrust_frac_min', upper_limit='thrust_frac_max',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_thrust_frac', method='sin')
)
climb.add_constant('thrust_frac_min', thrust_frac_min)
climb.add_constant('thrust_frac_max', thrust_frac_max)
climb.add_constant('eps_thrust_frac', 1e-1)

CL_min = 0.0
CL_max = 1.6
climb.add_inequality_constraint(
    'control', 'CL', lower_limit='CL_min', upper_limit='CL_max',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_CL', method='sin')
)
climb.add_constant('CL_min', CL_min)
climb.add_constant('CL_max', CL_max)
climb.add_constant('eps_CL', 1e-1)

# Atmospheric / Aerodynamic Properties
climb.add_expression('density', 'density0 * (temperature / temperature0) ** (-1 + g / (R_air * lapse_rate))')
climb.add_expression('temperature', 'temperature0 - lapse_rate * h')
climb.add_expression('qdyn', '0.5 * density * V ** 2')
climb.add_expression('mach', 'V / (gam_air * R_air * temperature) ** 0.5')
climb.add_expression('thrust_max', 'CT1 * (1 - h/CT2 + CT3 * h ** 2)')
climb.add_expression('CS', 'CS1 * (1 + V / CS2)')
climb.add_expression('CD', 'CD1 + CD2 * CL ** 2')

# Constants
climb.add_constant('s_ref', 122.6)
climb.add_constant('g', 9.81)
climb.add_constant('CT1', 141040.)
climb.add_constant('CT2', 14909.9)
climb.add_constant('CT3', 6.997e-10)
climb.add_constant('CD1', 0.0242)
climb.add_constant('CD2', 0.0469)
climb.add_constant('CS1', 1.055e-5)
climb.add_constant('CS2', 441.54)
climb.add_constant('R_air', 287.058)
climb.add_constant('temperature0', 288.15)
climb.add_constant('lapse_rate', 0.0065)
climb.add_constant('density0', 101325 / (287.058 * 288.15))  # rho = P / (R T)
# climb.add_constant('mu', 0.2857)
climb.add_constant('gam_air', 1.4)

speed_of_sound0 = (1.4 * 287.058 * 288.15) ** 0.5

# Boundary Values
climb.add_constant('h0', 3480.)
climb.add_constant('d0', 0.)
climb.add_constant('V0', 141.67)
climb.add_constant('gam0', 0.07)
climb.add_constant('mass0', 4.8e4)
# climb.add_constant('mass0', 7.6e4)

climb.add_constraint('initial', 't')
climb.add_constraint('initial', '(h - h0) / h_ref')
climb.add_constraint('initial', '(d - d0) / h_ref')
climb.add_constraint('initial', '(V - V0) / CAS_max')
climb.add_constraint('initial', '(gam - gam0) / gam_max')
climb.add_constraint('initial', '(mass - mass0) / mass0')

hf = 9144.
df = 150_000.
Vf = 191.0
gamf = 0.

climb.add_constant('hf', hf)
climb.add_constant('df', df)
climb.add_constant('Vf', Vf)
climb.add_constant('gamf', gamf)
climb.add_constant('h_ref', hf)

climb.add_constraint('terminal', '(h - hf) / h_ref')
climb.add_constraint('terminal', '(d - df) / h_ref')
climb.add_constraint('terminal', '(V - Vf) / CAS_max')
climb.add_constraint('terminal', '(gam - gamf) / gam_max')
# Terminal mass/time are free

# Constraint on FPA: FPA > 0
climb.add_inequality_constraint(
    'path', 'gam', lower_limit='gam_min - eps_gam * pi/180', upper_limit='gam_max',
    regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler('eps_gam', method='utm')
)
# climb.add_inequality_constraint(
#     'path', 'gam', lower_limit='gam_min - eps_gam * pi/180',
#     regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler('eps_gam', method='exterior')
# )
climb.add_constant('eps_gam', 1)
climb.add_constant('gam_min', -30 * np.pi / 180)  # Via continuation, drive to gam_min = 0
climb.add_constant('gam_max', 30 * np.pi / 180)

# Constraint on V: M < M_max
climb.add_inequality_constraint(
    'path', 'mach', lower_limit='0', upper_limit='mach_max',
    regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler('eps_mach', method='utm')
)
climb.add_constant('eps_mach', 1e-2)
# climb.add_constant('mach_max', 0.82)
climb.add_constant('mach_max', 2.0)
climb.add_constant('CAS_max', 0.82 * speed_of_sound0)

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_climb = giuseppe.problems.symbolic.SymDual(climb, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_climb, verbose=2, max_nodes=100, node_buffer=10)


def ctrl2reg(u: np.array, u_min: float, u_max: float) -> np.array:
    return np.arcsin((2*u - u_min - u_max) / (u_max - u_min))


def reg2ctrl(u_reg: np.array, u_min: float, u_max: float) -> np.array:
    return 0.5 * ((u_max - u_min) * np.sin(u_reg) + u_max + u_min)


def get_const_dict(_sol):
    _constants_dict = {}
    for key, val in zip(_sol.annotations.constants, _sol.k):
        _constants_dict[key] = val
    return _constants_dict


guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_climb,
    control=np.array((0.0, ctrl2reg(0.75, thrust_frac_min, thrust_frac_max))),
    t_span=2.5)

with open('mach_fpa_guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)
slope_const = (hf - seed_sol.x[0, 0]) / (df - seed_sol.x[1, 0])
gam_const = np.arctan(slope_const)
df_seed = (seed_sol.x[0, -1] - seed_sol.x[0, 0]) / slope_const

with open('mach_fpa_seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)

cont.add_linear_series(100, {'hf': seed_sol.x[0, -1] + 100, 'df': seed_sol.x[1, -1] + 500}, bisection=True)
cont.add_linear_series(100, {'gamf': gamf, 'hf': 5_000, 'df': 10_000}, bisection=True)
cont.add_linear_series(100, {'hf': hf, 'df': df, 'Vf': Vf}, bisection=True)
cont.add_logarithmic_series(50, {'eps_gam': 1e-3})
cont.add_linear_series(50, {'gam_min': -1. * np.pi/180, 'gam_max': 89. * np.pi/180})
cont.add_logarithmic_series(50, {'eps_gam': 1e-3})
cont.add_linear_series(100, {'gam_min': -1e-3 * np.pi/180})
# cont.add_logarithmic_series(50, {'eps_gam': 1e-4})
# cont.add_linear_series(100, {'gam_min': -1. * np.pi/180})

# cont.add_logarithmic_series(200, {'eps_gam': 1e-8, 'eps_mach': 1e-3},
#                             bisection=True)
# cont.add_logarithmic_series(250, {'eps_thrust_frac': 1e-3, 'eps_CL': 1e-3,
#                                   'eps_gam': 1e-4, 'eps_mach': 1e-4}, bisection=True)

sol_set = cont.run_continuation()

sol_set.save('mach_fpa_sol_set.data')
