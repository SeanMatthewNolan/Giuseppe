import pickle
import numpy as np

import matplotlib.pyplot as plt

with open('sol_set.data', 'rb') as file:
    sol = pickle.load(file)[-1]

re = 6.371e6
theta_cr = 600e3 / re

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

ax2 = fig.add_subplot(4, 1, 2)
ax2.plot(t, alpha * 180 / np.pi)
xlabel_2 = ax2.set_xlabel(r'$t$ [s]')
ylabel_2 = ax2.set_ylabel(r'$\alpha$ [deg]')

ax3 = fig.add_subplot(4, 1, 3)
ax3.plot(h, v)
xlabel_3 = ax3.set_xlabel(r'$t$ [s]')
ylabel_3 = ax3.set_ylabel(r'$v$ [m/s]')

ax4 = fig.add_subplot(4, 1, 4)
ax4.plot(t, gamma * 180 / np.pi)
xlabel_4 = ax4.set_xlabel(r'$t$ [s]')
ylabel_4 = ax4.set_ylabel(r'$\gamma$ [deg]')

fig.tight_layout()

fig_lam = plt.figure()
fig_lam.suptitle('Costates')

ax21 = fig_lam.add_subplot(4, 1, 1)
ax21.plot(t, sol.lam[0, :])
ax21.set_xlabel(r'$t$')
ax21.set_ylabel(r'$\lambda_h$')

ax22 = fig_lam.add_subplot(4, 1, 2)
ax22.plot(t, sol.lam[1, :])
ax22.set_xlabel(r'$t$')
ax22.set_ylabel(r'$\lambda_\theta$')

ax23 = fig_lam.add_subplot(4, 1, 3)
ax23.plot(t, sol.lam[2, :])
ax23.set_xlabel(r'$t$')
ax23.set_ylabel(r'$\lambda_v$')

ax24 = fig_lam.add_subplot(4, 1, 4)
ax24.plot(t, sol.lam[3, :])
ax24.set_xlabel(r'$t$')
ax24.set_ylabel(r'$\lambda_\gamma$')

fig_lam.tight_layout()

# Calculation of Sensor violation C4
c4 = (re * np.sin(theta) - (re + h) * np.sin(theta_cr) + 1/np.tan(theta_cr) * (re * np.cos(theta)) - (re + h) * np.cos(theta_cr))

fig_constraints = plt.figure()
fig_constraints.suptitle('Constraints')

ax31 = fig_constraints.add_subplot(1, 1, 1)
ax31.plot(t, c4 / re)
ax31.set_xlabel(r'$t$')
ax31.set_ylabel(r'$C_4 / r_e$')

fig_constraints.tight_layout()

# fig.savefig('brachistocrone.eps',
#             format='eps',
#             bbox_extra_artists=(title, xlabel_1, ylabel_1, xlabel_2, ylabel_2),
#             bbox_inches='tight')

plt.show()
