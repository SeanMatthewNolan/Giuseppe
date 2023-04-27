import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

DATA = 2
# PLOT = 'mach_fpa'
PLOT = 'unconstrained'

# BOUNDS = 'custom'
BOUNDS = 'default'

# --- DATA PROCESSING -----------------------------------------
if DATA == 0:
    with open(PLOT + '_guess.data', 'rb') as f:
        sol = pickle.load(f)
elif DATA == 1:
    with open(PLOT + '_seed_sol.data', 'rb') as f:
        sol = pickle.load(f)
else:
    with open(PLOT + '_sol_set.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]


# Convert regularized control back to control
def reg2ctrl(u_reg: np.array, u_min: float, u_max: float) -> np.array:
    return 0.5 * ((u_max - u_min) * np.sin(u_reg) + u_max + u_min)


def get_fpa_mach_bounds(_x, _const_dict):
    _temperature = _const_dict['temperature0'] - _const_dict['lapse_rate'] * _x[0, :]
    _speed_of_sound = (_const_dict['gam_air'] * _const_dict['R_air'] * _temperature) ** 0.5
    _mach_max = _const_dict['mach_max']

    _v_max = _mach_max * _speed_of_sound

    _fpa_min = _const_dict['gam_min'] - _const_dict['eps_gam'] * np.pi/180 + 0 * _x[3, :]
    return _mach_max, _v_max, _fpa_min


def get_default_bounds(_x, _const_dict):
    _temperature = _const_dict['temperature0'] - _const_dict['lapse_rate'] * _x[0, :]
    _speed_of_sound = (_const_dict['gam_air'] * _const_dict['R_air'] * _temperature) ** 0.5
    _mach_max = 0.82

    _v_max = _mach_max * _speed_of_sound

    _fpa_min = 0.0 + 0 * _x[3, :]
    return _mach_max, _v_max, _fpa_min


def get_mach(_x, _const_dict):
    _temperature = _const_dict['temperature0'] - _const_dict['lapse_rate'] * _x[0, :]
    _speed_of_sound = (_const_dict['gam_air'] * _const_dict['R_air'] * _temperature) ** 0.5
    _mach = _x[2, :] / _speed_of_sound
    return _mach


constants_dict = {}
for key, val in zip(sol.annotations.constants, sol.k):
    constants_dict[key] = val

thrust_frac = reg2ctrl(sol.u[0, :], constants_dict['thrust_frac_min'], constants_dict['thrust_frac_max'])
CL = reg2ctrl(sol.u[1, :], constants_dict['CL_min'], constants_dict['CL_max'])

# --- PLOTTING -----------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
tlab = 'Time [s]'

# PLOT STATES
ylabs = (r'$h$ [m]', r'$d$ [m]', r'$V$ [m/s]', r'$\gamma$ [deg]', r'$m$ [kg]')
mult = np.array((1., 1., 1., 180/np.pi, 1.))
np.dtype(float)
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(2, 3, idx + 1))
    ax = axes_states[-1]
    ax.ticklabel_format(style='plain')

    ax.grid()
    ax.set_xlabel(tlab)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state * mult[idx])

if BOUNDS == 'custom':
    mach_max, v_max, fpa_min = get_fpa_mach_bounds(sol.x, constants_dict)
else:
    mach_max, v_max, fpa_min = get_default_bounds(sol.x, constants_dict)

axes_states[2].plot(sol.t, v_max * mult[2], 'k--')
axes_states[3].plot(sol.t, fpa_min * mult[3], 'k--')

fig_states.suptitle(rf'$J(\alpha = {constants_dict["frac_time_cost"]}) = {sol.cost:.2f}$, '
                    + rf'$m_0 - m_f = {sol.x[-1, 0] - sol.x[-1, -1]:.2f}$, $t_f = {sol.t[-1]:.2f}$')
fig_states.tight_layout()

# PLOT CONTROL
control_sets = (thrust_frac, CL)
ulabs = (r'$\epsilon_T$', r'$C_L$')

fig_controls = plt.figure()
axes_controls = []

for idx, u in enumerate(control_sets):
    axes_controls.append(fig_controls.add_subplot(1, 2, idx + 1))
    ax = axes_controls[-1]
    ax.ticklabel_format(style='plain')
    ax.grid()
    ax.set_xlabel(tlab)
    ax.set_ylabel(ulabs[idx])
    ax.plot(sol.t, u)

fig_controls.tight_layout()

# PLOT COSTATES
ylabs = (r'$\lambda_h$', r'$\lambda_d$', r'$\lambda_V$', r'$\lambda_\gamma$', r'$\lambda_m$')
np.dtype(float)
fig_costates = plt.figure()
axes_costates = []

for idx, costate in enumerate(list(sol.lam)):
    axes_costates.append(fig_costates.add_subplot(2, 3, idx + 1))
    ax = axes_costates[-1]
    ax.ticklabel_format(style='plain')

    ax.grid()
    ax.set_xlabel(tlab)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, costate)

fig_costates.tight_layout()

# PLOT FROM PAPER
mach = get_mach(sol.x, constants_dict)
constraint_sets = (fpa_min - sol.x[3, :], mach - mach_max)
clabs = (r'$\gamma_{min} - \gamma$ [deg]', r'$M - M_{max}$ [m/s]')
cmult = (180 / np.pi, 1)

fig_paper = plt.figure()
axes_paper = []

for idx, (c, u) in enumerate(zip(constraint_sets, control_sets)):
    axes_paper.append(fig_paper.add_subplot(2, 2, idx + 1))
    ax = axes_paper[-1]
    ax.ticklabel_format(style='plain')
    ax.grid()
    ax.set_xlabel(tlab)
    ax.set_ylabel(clabs[idx])
    ax.plot(sol.t, c * cmult[idx])
    ax.plot(sol.t[((0, -1),)], np.zeros((2,)), 'k--')

    axes_paper.append(fig_paper.add_subplot(2, 2, idx + 3))
    ax = axes_paper[-1]
    ax.ticklabel_format(style='plain')
    ax.grid()
    ax.set_xlabel(tlab)
    ax.set_ylabel(ulabs[idx])
    ax.plot(sol.t, u)

fig_paper.tight_layout()

np.savetxt(PLOT + '_txu.csv', np.vstack((sol.t.reshape((1, -1)), sol.x, thrust_frac, CL)), delimiter=',')

plt.show()
