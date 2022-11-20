import pickle

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

MED_FIG_SIZE = (6.5, 5)
SML_FIG_SIZE = (6.5, 3)

with open('sol_set.data', 'rb') as file:
    sol_set = pickle.load(file)

cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
gradient = mpl.colormaps['viridis'].colors
grad_idcs = np.int32(np.ceil(np.linspace(255, 0, len(sol_set))))


def cols_gradient(n):
    return gradient[grad_idcs[n]]


fig = plt.figure(figsize=SML_FIG_SIZE)
# fig.suptitle('Brachistochrone')

ax1 = fig.add_subplot(2, 1, 1)
ax1.grid()
ax1.set_xlabel(r'$x$ [ft]')
ax1.set_ylabel(r'$y$ [ft]')

ax2 = fig.add_subplot(2, 1, 2)
ax2.grid()
ax2.set_xlabel(r'$t$ [s]')
ax2.set_ylabel(r'$\theta$ [deg]')

fig_lam = plt.figure(figsize=MED_FIG_SIZE)
# fig_lam.suptitle('Costates')

ax21 = fig_lam.add_subplot(3, 1, 1)
ax21.grid()
# ax21.set_xlabel(r'$t$ [s]')
ax21.set_ylabel(r'$\lambda_x$ [s/ft]')

ax22 = fig_lam.add_subplot(3, 1, 2)
ax22.grid()
# ax22.set_xlabel(r'$t$ [s]')
ax22.set_ylabel(r'$\lambda_y$ [s/ft]')

ax23 = fig_lam.add_subplot(3, 1, 3)
ax23.grid()
ax23.set_xlabel(r'$t$ [s]')
ax23.set_ylabel(r'$\lambda_v$ [s$^2$/ft]')

for idx, sol in enumerate(sol_set):
    ax1.plot(sol.x[0, :], sol.x[1, :], color=cols_gradient(idx))
    ax2.plot(sol.t, sol.u[0, :] * 180 / np.pi, color=cols_gradient(idx))

    ax21.plot(sol.t, sol.lam[0, :], color=cols_gradient(idx))
    ax22.plot(sol.t, sol.lam[1, :], color=cols_gradient(idx))
    ax23.plot(sol.t, sol.lam[2, :], color=cols_gradient(idx))

fig.tight_layout()
fig_lam.tight_layout()

fig.savefig('brachistocrone_continuations.eps',
            format='eps',
            bbox_inches='tight')

fig_lam.savefig('brachistocrone_continuation_costates.eps',
                format='eps',
                bbox_inches='tight')

plt.show()
