import pickle
import numpy as np

import matplotlib.pyplot as plt

with open('sol_set.data', 'rb') as file:
    sol = pickle.load(file)[-1]

fig = plt.figure(figsize=(6.5, 5))
title = fig.suptitle('Glide Vehicle')

ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(sol.x[1, :] * 180/np.pi, sol.x[0, :] / 1000)
xlabel_1 = ax1.set_xlabel(r'$\theta$ [deg]')
ylabel_1 = ax1.set_ylabel(r'$h$ [km]')

ax2 = fig.add_subplot(4, 1, 2)
ax2.plot(sol.t, sol.u[0, :] * 180 / 3.14159)
xlabel_2 = ax2.set_xlabel(r'$t$ [s]')
ylabel_2 = ax2.set_ylabel(r'$\alpha$ [deg]')

ax3 = fig.add_subplot(4, 1, 3)
ax3.plot(sol.t, sol.x[2, :])
xlabel_3 = ax3.set_xlabel(r'$t$ [s]')
ylabel_3 = ax3.set_ylabel(r'$v$ [m/s]')

ax4 = fig.add_subplot(4, 1, 4)
ax4.plot(sol.t, sol.x[3, :] * 180 / 3.14159)
xlabel_4 = ax4.set_xlabel(r'$t$ [s]')
ylabel_4 = ax4.set_ylabel(r'$\gamma$ [deg]')

fig.tight_layout()

fig_lam = plt.figure()
fig_lam.suptitle('Costates')

ax21 = fig_lam.add_subplot(3, 1, 1)
ax21.plot(sol.t, sol.lam[0, :])
ax21.set_xlabel(r'$t$')
ax21.set_ylabel(r'$\lambda_x$')

ax22 = fig_lam.add_subplot(3, 1, 2)
ax22.plot(sol.t, sol.lam[1, :])
ax22.set_xlabel(r'$t$')
ax22.set_ylabel(r'$\lambda_y$')

ax23 = fig_lam.add_subplot(3, 1, 3)
ax23.plot(sol.t, sol.lam[2, :])
ax23.set_xlabel(r'$t$')
ax23.set_ylabel(r'$\lambda_v$')

fig_lam.tight_layout()

# fig.savefig('brachistocrone.eps',
#             format='eps',
#             bbox_extra_artists=(title, xlabel_1, ylabel_1, xlabel_2, ylabel_2),
#             bbox_inches='tight')

plt.show()
