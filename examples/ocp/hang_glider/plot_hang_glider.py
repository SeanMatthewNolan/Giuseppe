import pickle
import matplotlib.pyplot as plt
import numpy as np

DATA = 2

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

constants_dict = {}
for key, val in zip(sol.annotations.constants, sol.k):
    constants_dict[key] = val


u = 0.5 * ((constants_dict['CL_max'] - constants_dict['CL_min']) * np.sin(sol.u)
           + constants_dict['CL_max'] + constants_dict['CL_min'])

# PLOT STATES
ylabs = (r'$x$', r'$y$', r'$v_x$', r'$v_y$')
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state)

fig_states.tight_layout()

# PLOT U
ylabs = (r'$C_L$',)
fig_u = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list(u)):
    axes_u.append(fig_u.add_subplot(1, 1, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl)

fig_u.tight_layout()

# PLOT COSTATES
ylabs = (r'$\lambda_{x}$', r'$\lambda_{y}$', r'$\lambda_{v_x}$', r'$\lambda_{v_y}$')
fig_costates = plt.figure()
axes_costates = []

for idx, costate in enumerate(list(sol.lam)):
    axes_costates.append(fig_costates.add_subplot(2, 2, idx + 1))
    ax = axes_costates[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, costate)

fig_costates.tight_layout()

np.savetxt('txu_Solution.csv', np.vstack((sol.t.reshape((1, -1)), sol.x, u,)), delimiter=',')

plt.show()
