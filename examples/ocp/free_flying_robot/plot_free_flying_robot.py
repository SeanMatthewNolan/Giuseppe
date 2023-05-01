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

eps_u = sol.k[15]
u_min = sol.k[16]
u_max = sol.k[17]
# u = (u_max - u_min) / np.pi * np.arctan(sol.u / eps_u) + (u_max + u_min)/2
u = 0.5 * ((u_max - u_min) * np.sin(sol.u) + u_max + u_min)

dJ_dt = np.sum(u, 0)

t1 = u[0, :] - u[1, :]
t2 = u[2, :] - u[3, :]

# PLOT STATES
ylabs = (r'$x$', r'$y$', r'$v_x$', r'$v_y$', r'$\theta$', r'$\omega$')
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(2, 3, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state)

fig_states.tight_layout()

# PLOT T
tsets = (t1, t2)
ylabs = (r'$T_1$', r'$T_2$')

fig_T = plt.figure()
axes_T = []

for idx, t in enumerate(tsets):
    axes_T.append(fig_T.add_subplot(1, 2, idx + 1))
    ax = axes_T[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, t)

fig_T.tight_layout()

# PLOT U
ylabs = (r'$u_1$', r'$u_2$', r'$u_3$', r'$u_4$')
fig_u = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list(u)):
    axes_u.append(fig_u.add_subplot(2, 2, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl)

fig_u.tight_layout()

# PLOT COST

fig_cost = plt.figure()
ax_cost = fig_cost.add_subplot(111)
ax_cost.grid()
ax_cost.plot(sol.t, dJ_dt)
ax_cost.set_xlabel('Time [s]')
ax_cost.set_ylabel(r'Path Cost $|T_1| + |T_2|$')
ax_cost.set_title(f'Total Cost = {sol.cost}')

np.savetxt('txuTSolution.csv', np.vstack((sol.t.reshape((1, -1)), sol.x, u, np.array((t1, t2)))), delimiter=',')

plt.show()
