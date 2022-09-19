import pickle

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
from atmosphere1976 import Atmosphere1976

DATA = 0

if DATA == 0:
    with open('sol_set.data', 'rb') as file:
        sol = pickle.load(file)[-1]
elif DATA == 1:
    with open('seed.data', 'rb') as file:
        sol = pickle.load(file)
elif DATA == 2:
    with open('guess.data', 'rb') as file:
        sol = pickle.load(file)


# FIGURE 1 (MAIN)
fig = plt.figure(figsize=(6.5, 5))
title = fig.suptitle('Space Shuttle Crossrange')

ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(sol.x[1, :] * 180/np.pi, sol.x[0, :] / 1000)
xlabel_1 = ax1.set_xlabel(r'$\phi$ [deg]')
ylabel_1 = ax1.set_ylabel(r'$h$ [km]')

ax2 = fig.add_subplot(4, 1, 2)
ax2.plot(sol.t, sol.u[0, :] * 180 / 3.14159, label=r'$\alpha$')
ax2.plot(sol.t, sol.u[1, :] * 180 / 3.14159, label=r'$\beta$')
xlabel_2 = ax2.set_xlabel(r'$t$ [s]')
ylabel_2 = ax2.set_ylabel(r'$u$ [deg]')
ax2.legend()

ax3 = fig.add_subplot(4, 1, 3)
ax3.plot(sol.x[1, :], sol.x[2, :])
xlabel_3 = ax3.set_xlabel(r'$\phi$ [s]')
ylabel_3 = ax3.set_ylabel(r'$\theta$ [m/s]')

ax4 = fig.add_subplot(4, 1, 4)
ax4.plot(sol.t, sol.x[4, :] * 180 / 3.14159)
xlabel_4 = ax4.set_xlabel(r'$t$ [s]')
ylabel_4 = ax4.set_ylabel(r'$\gamma$ [deg]')

fig.tight_layout()

# FIGURE 2 (STATES)
fig_states = plt.figure()
fig_states.suptitle('States')

ax21 = fig_states.add_subplot(2, 4, 1)
ax21.plot(sol.t, sol.x[0, :])
ax21.set_xlabel(r'$t$')
ax21.set_ylabel(r'$h$')

ax22 = fig_states.add_subplot(2, 4, 2)
ax22.plot(sol.t, sol.x[1, :] / np.pi * 180)
ax22.set_xlabel(r'$t$')
ax22.set_ylabel(r'$\phi$')

ax23 = fig_states.add_subplot(2, 4, 3)
ax23.plot(sol.t, sol.x[2, :] / np.pi * 180)
ax23.set_xlabel(r'$t$')
ax23.set_ylabel(r'$\theta$')

ax24 = fig_states.add_subplot(2, 4, 4)
ax24.plot(sol.t, sol.x[3, :])
ax24.set_xlabel(r'$t$')
ax24.set_ylabel(r'$v$')

ax24 = fig_states.add_subplot(2, 4, 5)
ax24.plot(sol.t, sol.x[4, :] / np.pi * 180)
ax24.set_xlabel(r'$t$')
ax24.set_ylabel(r'$\gamma$')

ax24 = fig_states.add_subplot(2, 4, 6)
ax24.plot(sol.t, sol.x[5, :] / np.pi * 180)
ax24.set_xlabel(r'$t$')
ax24.set_ylabel(r'$\psi$')

ax25 = fig_states.add_subplot(2, 4, 7)
ax25.plot(sol.t, sol.u[0, :] / np.pi * 180)
ax25.set_xlabel(r'$t$')
ax25.set_ylabel(r'$\alpha$')

ax26 = fig_states.add_subplot(2, 4, 8)
ax26.plot(sol.t, sol.u[1, :] / np.pi * 180)
ax26.set_xlabel(r'$t$')
ax26.set_ylabel(r'$\beta$')

fig_states.tight_layout()

fig.tight_layout()

# FIGURE 3 (COSTATES)
fig_lam = plt.figure()
fig_lam.suptitle('Costates')

ax21 = fig_lam.add_subplot(2, 3, 1)
ax21.plot(sol.t, sol.lam[0, :])
ax21.set_xlabel(r'$t$')
ax21.set_ylabel(r'$\lambda_h$')

ax22 = fig_lam.add_subplot(2, 3, 2)
ax22.plot(sol.t, sol.lam[1, :])
ax22.set_xlabel(r'$t$')
ax22.set_ylabel(r'$\lambda_\phi$')

ax23 = fig_lam.add_subplot(2, 3, 3)
ax23.plot(sol.t, sol.lam[2, :])
ax23.set_xlabel(r'$t$')
ax23.set_ylabel(r'$\lambda_\theta$')

ax24 = fig_lam.add_subplot(2, 3, 4)
ax24.plot(sol.t, sol.lam[3, :])
ax24.set_xlabel(r'$t$')
ax24.set_ylabel(r'$\lambda_v$')

ax24 = fig_lam.add_subplot(2, 3, 5)
ax24.plot(sol.t, sol.lam[4, :])
ax24.set_xlabel(r'$t$')
ax24.set_ylabel(r'$\lambda_\gamma$')

ax24 = fig_lam.add_subplot(2, 3, 6)
ax24.plot(sol.t, sol.lam[5, :])
ax24.set_xlabel(r'$t$')
ax24.set_ylabel(r'$\lambda_\psi$')

fig_lam.tight_layout()

# FIGURE 4 (ATMOSPHERE)
re = sol.k
atm = Atmosphere1976(use_metric=False)
rho_0 = 0.002378  # slug/ft^3
h_ref = 23_800  # ft

dens_exp = rho_0 * np.exp(-sol.x[0, :] / h_ref)
dens_exp_deriv = -rho_0 / h_ref * np.exp(-sol.x[0, :] / h_ref)

h_sx = ca.SX.sym('h', 1)
_, __, dens_expr = atm.get_sx_atm_expr(h_sx)
dens_deriv_expr = ca.jacobian(dens_expr, h_sx)

dens_ca_func = ca.Function('rho', (h_sx,), (dens_expr,), ('h',), ('rho',))
dens_deriv_ca_func = ca.Function('drho_dh', (h_sx,), (dens_deriv_expr,), ('h',), ('drho_dh',))

dens_cond = np.empty(shape=sol.x[0, :].shape)
dens_cond_deriv = np.empty(shape=sol.x[0, :].shape)
layer_cond = list()

for i, h in enumerate(sol.x[0, :]):
    dens_cond[i] = dens_ca_func(h)
    dens_cond_deriv[i] = dens_deriv_ca_func(h)
    layer_cond.append(atm.layer(h))

layer_cond = np.array(layer_cond)

fig_atm = plt.figure(figsize=(6.5, 5))
fig_atm.suptitle('Atmosphere')

ax41 = fig_atm.add_subplot(121)
ax42 = fig_atm.add_subplot(122)

ax41.plot(sol.t, dens_exp * 100_000, label='Exponential')
ax42.plot(sol.t, dens_exp_deriv * 1e9, label='Exponential')
for layer in atm.layer_names:
    layer_idcs = np.where(layer_cond == layer)
    if len(layer_idcs[0]) > 0:
        ax41.plot(sol.t[layer_idcs], dens_cond[layer_idcs] * 100_000, label=layer)
        ax42.plot(sol.t[layer_idcs], dens_cond_deriv[layer_idcs] * 1e9, label=layer)

ax41.grid()
ax42.grid()

ax41.set_xlabel(r'$t$')
ax42.set_xlabel(r'$t$')
ax41.set_ylabel(r'$\rho$ [slug / 100,000 ft^3]')
ax42.set_ylabel(r'$\dfrac{d\rho}{dh}$ [slug / 10^9 ft^4}')

ax41.legend()

# fig.savefig('brachistocrone.eps',
#             format='eps',
#             bbox_extra_artists=(title, xlabel_1, ylabel_1, xlabel_2, ylabel_2),
#             bbox_inches='tight')

plt.show()
