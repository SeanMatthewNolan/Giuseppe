import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

with open('sol_set.data', 'rb') as file:
    sols = pickle.load(file)
    sol = sols[-1]

re = 6.371e6
theta_cr = 2 * 600e3 / re

t = sol.t

h = sol.x[0, :]  # m
theta = sol.x[1, :]  # rad
v = sol.x[2, :]  # m/s
gamma = sol.x[3, :]  # rad

alpha = sol.u[0, :]  # rad


fig = plt.figure(figsize=(6.5, 5))
title = fig.suptitle('Glide Vehicle')

ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(theta * 180/np.pi, h / 1000)
xlabel_1 = ax1.set_xlabel(r'$\theta$ [deg]')
ylabel_1 = ax1.set_ylabel(r'$h$ [km]')
ax1.grid()

ax2 = fig.add_subplot(4, 1, 2)
ax2.plot(t, alpha * 180 / np.pi)
xlabel_2 = ax2.set_xlabel(r'$t$ [s]')
ylabel_2 = ax2.set_ylabel(r'$\alpha$ [deg]')
ax2.grid()

ax3 = fig.add_subplot(4, 1, 3)
ax3.plot(h / 1000, v)
xlabel_3 = ax3.set_xlabel(r'$h$ [km]')
ylabel_3 = ax3.set_ylabel(r'$v$ [m/s]')
ax3.grid()

ax4 = fig.add_subplot(4, 1, 4)
ax4.plot(t, gamma * 180 / np.pi)
xlabel_4 = ax4.set_xlabel(r'$t$ [s]')
ylabel_4 = ax4.set_ylabel(r'$\gamma$ [deg]')
ax4.grid()

fig.tight_layout()

fig_lam = plt.figure()
fig_lam.suptitle('Costates')

ax21 = fig_lam.add_subplot(4, 1, 1)
ax21.plot(t, sol.lam[0, :])
ax21.set_xlabel(r'$t$')
ax21.set_ylabel(r'$\lambda_h$')
ax21.grid()

ax22 = fig_lam.add_subplot(4, 1, 2)
ax22.plot(t, sol.lam[1, :])
ax22.set_xlabel(r'$t$')
ax22.set_ylabel(r'$\lambda_\theta$')
ax22.grid()

ax23 = fig_lam.add_subplot(4, 1, 3)
ax23.plot(t, sol.lam[2, :])
ax23.set_xlabel(r'$t$')
ax23.set_ylabel(r'$\lambda_v$')
ax23.grid()

ax24 = fig_lam.add_subplot(4, 1, 4)
ax24.plot(t, sol.lam[3, :])
ax24.set_xlabel(r'$t$')
ax24.set_ylabel(r'$\lambda_\gamma$')
ax24.grid()

fig_lam.tight_layout()

# # Calculation of Sensor violation C4
# c4 = -(re * np.sin(theta) - (re + h) * np.sin(theta_cr) + 1/np.tan(theta_cr) * (re * np.cos(theta)) - (re + h) * np.cos(theta_cr))
#
# fig_constraints = plt.figure()
# fig_constraints.suptitle('Constraints')
#
# ax31 = fig_constraints.add_subplot(1, 1, 1)
# ax31.plot(t, c4 / re)
# ax31.set_xlabel(r'$t$')
# ax31.set_ylabel(r'$C_4 / r_e$')
# ax31.grid()
#
# fig_constraints.tight_layout()

cmap = cm.get_cmap('viridis', len(sols)).reversed()

fig_continuations = plt.figure(figsize=(6.5, 5))
title_continuations = fig_continuations.suptitle('Continuations of Solution')
ax41 = fig_continuations.add_subplot(2, 1, 1)
ax42 = fig_continuations.add_subplot(2, 1, 2)
for i, continuation in enumerate(sols):
    ax41.plot(continuation.x[1, :] * 180/np.pi, continuation.x[0, :] / 1000, c=cmap(i))
    ax42.plot(continuation.t, continuation.u[0, :] * 180/np.pi, c=cmap(i))

xlabel_41 = ax41.set_xlabel(r'$\theta$ [deg]')
ylabel_41 = ax41.set_ylabel(r'$h$ [km]')
ax41.grid()

xlabel_42 = ax42.set_xlabel(r'$t$ [s]')
ylabel_42 = ax42.set_ylabel(r'$\alpha$ [deg]')
ax42.grid()

# fig.savefig('brachistocrone.eps',
#             format='eps',
#             bbox_extra_artists=(title, xlabel_1, ylabel_1, xlabel_2, ylabel_2),
#             bbox_inches='tight')

plt.show()
