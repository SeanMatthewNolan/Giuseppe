import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

DATA = 1

# --- DATA PROCESSING -----------------------------------------
if DATA == 0:
    with open('guess.data', 'rb') as f:
        sol = pickle.load(f)
elif DATA == 1:
    with open('seed_sol.data', 'rb') as f:
        sol = pickle.load(f)
else:
    with open('sol_set.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]

# --- PLOTTING -----------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
tlab = 'Time [s]'

# PLOT STATES
ylabs = (r'$x$ [ft]', r'$y$ [ft]', r'$z$ [ft]', r'$V$ [ft/s]', r'$\gamma$ [deg]', r'$\psi$ [deg]')
mult = np.array((1., 1., 1., 1., 180./np.pi, 180./np.pi))
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

fig_states.suptitle(rf'$\beta$ = {sol.p[0]}')
fig_states.tight_layout()

# PLOT CONTROL
ulabs = (r'$C_L$', r'$\sigma$ [deg]')
umult = np.array((1.0, 180./np.pi))

fig_controls = plt.figure()
axes_controls = []

for idx, u in enumerate(list(sol.u)):
    axes_controls.append(fig_controls.add_subplot(1, 2, idx + 1))
    ax = axes_controls[-1]
    ax.ticklabel_format(style='plain')
    ax.grid()
    ax.set_xlabel(tlab)
    ax.set_ylabel(ulabs[idx])
    ax.plot(sol.t, u * umult[idx])

fig_controls.tight_layout()

# PLOT COSTATES
ylabs = (r'$\lambda_x$', r'$\lambda_y$', r'$\lambda_z$', r'$\lambda_V$', r'$\lambda_\gamma$', r'$\lambda_\psi$',
         r'$\lambda_\beta$', r'$\lambda_{V_0}$', r'$\lambda_{\gamma_0}$', r'$\lambda_{\psi_0}$', )
np.dtype(float)
fig_costates = plt.figure()
axes_costates = []

for idx, costate in enumerate(list(sol.lam)):
    axes_costates.append(fig_costates.add_subplot(2, 5, idx + 1))
    ax = axes_costates[-1]
    ax.ticklabel_format(style='plain')

    ax.grid()
    ax.set_xlabel(tlab)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, costate)

    lam_min = np.min(costate)
    lam_max = np.max(costate)

    if np.max(costate) - np.min(costate) < 1e-6:
        ax.set_ylim((lam_min - 1., lam_max + 1.))

fig_costates.tight_layout()

plt.show()
