import pickle

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from giuseppe.utils.examples.atmosphere1976 import Atmosphere1976
# from space_shuttle_crossrange_conditional_atm import adiff_bvp

SMALL_FIGSIZE = (6.5, 3)
MED_FIG_SIZE = (6.5, 5)
BIG_FIG_SIZE = (6.5, 6.5)
T_LAB = r'$t$ [s]'

r2d = 180 / np.pi
d2r = np.pi / 180

with open('sol_set.data', 'rb') as file:
    sol = pickle.load(file)[-1]
# with open('sol_set_25_000.data', 'rb') as file:
#     sol_25k = pickle.load(file)[-1]
with open('sol_set_conditional.data', 'rb') as file:
    sol_cond = pickle.load(file)[-1]
# with open('sol_set_conditional_25_000.data', 'rb') as file:
#     sol_cond_25k = pickle.load(file)[-1]

# Create Dicts
exp_dict = {}
cond_dict = {}

for key, exp_val, cond_val in zip(sol.annotations.constants, sol.k, sol_cond.k):
    exp_dict[key] = exp_val
    cond_dict[key] = cond_val
for key, exp_x_val, exp_lam_val, cond_x_val, cond_lam_val in zip(
        sol.annotations.states, list(sol.x), list(sol.lam), list(sol_cond.x), list(sol_cond.lam)
):
    exp_dict[key] = exp_x_val
    exp_dict['lam_' + key] = exp_lam_val
    cond_dict[key] = cond_x_val
    cond_dict['lam_' + key] = cond_lam_val
for key, exp_val, cond_val in zip(sol.annotations.controls, list(sol.u), list(sol_cond.u)):
    exp_dict[key] = exp_val
    cond_dict[key] = cond_val
exp_dict[sol.annotations.independent] = sol.t
cond_dict[sol.annotations.independent] = sol_cond.t

re = exp_dict['re']
mu = exp_dict['mu']
g0 = mu / re**2
atm = Atmosphere1976(use_metric=False, earth_radius=re, gravity=g0, boundary_thickness=1000.)

rho_0 = exp_dict['rho_0']  # slug/ft^3
h_ref = exp_dict['h_ref'] # ft

dens_exp = rho_0 * np.exp(-sol.x[0, :] / h_ref)
dens_exp_deriv = -dens_exp / h_ref

h_sx = ca.SX.sym('h', 1)
_, __, dens_expr = atm.get_ca_atm_expr(h_sx)
dens_deriv_expr = ca.jacobian(dens_expr, h_sx)

dens_ca_func = ca.Function('rho', (h_sx,), (dens_expr,), ('h',), ('rho',))
dens_deriv_ca_func = ca.Function('drho_dh', (h_sx,), (dens_deriv_expr,), ('h',), ('drho_dh',))

dens_cond = np.empty(shape=sol_cond.x[0, :].shape)
dens_cond_deriv = np.empty(shape=sol_cond.x[0, :].shape)
layer_cond = list()

for i, h in enumerate(sol_cond.x[0, :]):
    dens_cond[i] = dens_ca_func(h)
    dens_cond_deriv[i] = dens_deriv_ca_func(h)
    layer_cond.append(atm.layer(h))

layer_cond = np.array(layer_cond)

qdyn_exp = 0.5 * dens_exp * exp_dict['v'] ** 2
qdyn_cond = 0.5 * dens_cond * cond_dict['v'] ** 2

# FIGURE 1 (STATES)
fig1 = plt.figure(figsize=MED_FIG_SIZE)

plot_nums = (1, 3, 5, 2, 4, 6)
ylabs = (r'$h$ [100,000 ft]', r'$\phi$ [deg]', r'$\theta$ [deg]',
          r'$V$ [1,000 ft/s]', r'$\gamma$ [deg]', r'$\psi$ [deg]')
mult = (1e-5, r2d, r2d, 1e-3, r2d, r2d)
ax1 = []
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for idx, plot_num in enumerate(plot_nums):
    ax1.append(fig1.add_subplot(3, 2, plot_num))
    ax1[idx].grid()
    ax1[idx].plot(sol.t, sol.x[idx, :] * mult[idx], color=colors[0], label='Exp.')
    ax1[idx].plot(sol_cond.t, sol_cond.x[idx, :] * mult[idx], zorder=0, color=colors[1], label='Cond.')
    ax1[idx].set_ylabel(ylabs[idx])

ax1[4].legend()
ax1[-1].set_xlabel(T_LAB)
ax1[2].set_xlabel(T_LAB)

fig1.tight_layout()

# FIGURE 2 (CONTROL HISTORIES)
fig2 = plt.figure(figsize=SMALL_FIGSIZE)

ax21 = fig2.add_subplot(211)
ax21.grid()
ax21.plot(sol.t, sol.u[0, :] * r2d, label='Exp.')
ax21.plot(sol_cond.t, sol_cond.u[0, :] * r2d, zorder=0, label='Cond.')
ax21.set_ylabel(r'$\alpha$ [deg]')
ax21.legend()

ax22 = fig2.add_subplot(212)
ax22.grid()
ax22.plot(sol.t, sol.u[1, :] * r2d, label='Exp.')
ax22.plot(sol_cond.t, sol_cond.u[1, :] * r2d, zorder=0, label='Cond.')
ax22.set_ylabel(r'$\sigma$ [deg]')
ax22.set_xlabel(T_LAB)

fig2.tight_layout()

# FIGURE 3 (ATMOSPHERE)
fig3 = plt.figure(figsize=SMALL_FIGSIZE)

ax31 = fig3.add_subplot(211)
ax31.grid()
ax31.plot(sol.t, dens_exp, label='Exp.')
ax31.plot(sol_cond.t, dens_cond,  zorder=0, label='Cond.')
ax31.set_ylabel(r'$\rho$ [slug / ft$^3$]')
ax31.set_yscale('log')

ax32 = fig3.add_subplot(212)
ax32.grid()
ax32.plot(sol.t, -dens_exp_deriv, label='Exp.')
ax32.plot(sol_cond.t, -dens_cond_deriv,  zorder=0, label='Cond.')
ax32.set_ylabel(r'$-\dfrac{d\rho}{dh}$ [slug / ft$^4$]')
ax32.set_yscale('log')

ax32.legend()

fig3.tight_layout()

# FIGURE 4 (COSTATES)
fig4 = plt.figure(figsize=MED_FIG_SIZE)

plot_nums = (1, 3, 5, 2, 4, 6)
ylabs = (r'$\lambda_h$ [deg/ft-s]', r'$\lambda_\phi$ [s$^{-1}$]', r'$\lambda_\theta$ [s$^{-1}$]',
         r'$\lambda_V$ [deg/ft]', r'$\lambda_\gamma$ [s$^{-1}$]', r'$\lambda_\psi$ [s$^{-1}$]')
mult = (r2d, 1, 1, r2d, 1, 1)
ax4 = []
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for idx, plot_num in enumerate(plot_nums):
    ax4.append(fig4.add_subplot(3, 2, plot_num))
    ax4[idx].grid()
    ax4[idx].plot(sol.t, sol.lam[idx, :] * mult[idx], color=colors[0], label='Exp.')
    ax4[idx].plot(sol_cond.t, sol_cond.lam[idx, :] * mult[idx], zorder=0, color=colors[1], label='Cond.')
    ax4[idx].set_ylabel(ylabs[idx])
    ylim = ax4[idx].get_ylim()
    ax4[idx].set_ylim(bottom=min(ylim[0], -1e-6), top=max(ylim[-1], 1e-6))

ax4[4].legend()
ax4[-1].set_xlabel(T_LAB)
ax4[2].set_xlabel(T_LAB)

fig4.tight_layout()

# FIGURE 5 (PAPER)
xdata_exp = (exp_dict['φ'], exp_dict['v'], exp_dict['t'], exp_dict['t'])
ydata_exp = (exp_dict['θ'], exp_dict['h'], qdyn_exp, exp_dict['α'])

xdata_cond = (cond_dict['φ'], cond_dict['v'], cond_dict['t'], cond_dict['t'])
ydata_cond = (cond_dict['θ'], cond_dict['h'], qdyn_cond, cond_dict['α'])

ylabs = (r'$\theta$ [deg]', r'$h$ [1,000 ft]', r'$Q_{\infty}$ [psf]', r'$\alpha$ [deg]')
xlabs = (r'$\phi$ [deg]', r'$V$ [1,000 ft/s]', T_LAB, T_LAB)
xmult = (r2d, 1e-3, 1, 1)
ymult = (r2d, 1e-3, 1, r2d)
plot_nums = (1, 2, 3, 4)

fig5 = plt.figure(figsize=MED_FIG_SIZE)
ax5 = []

for idx, plot_num in enumerate(plot_nums):
    ax5.append(fig5.add_subplot(2, 2, plot_num))
    ax5[idx].grid()
    ax5[idx].plot(xdata_exp[idx] * xmult[idx], ydata_exp[idx] * ymult[idx], 'k-', label='Exponential')
    ax5[idx].plot(xdata_cond[idx] * xmult[idx], ydata_cond[idx] * ymult[idx], 'k--', label='Conditional')
    ax5[idx].set_ylabel(ylabs[idx])
    ax5[idx].set_xlabel(xlabs[idx])

ax5[-1].legend()
fig5.tight_layout()

# SAVE FIGURES
fig1.savefig('space_shuttle_states.eps',
             format='eps',
             bbox_inches='tight')

fig2.savefig('space_shuttle_control_history.eps',
             format='eps',
             bbox_inches='tight')

fig3.savefig('space_shuttle_density.eps',
             format='eps',
             bbox_inches='tight')

fig4.savefig('space_shuttle_costates.eps',
             format='eps',
             bbox_inches='tight')

fig5.savefig('space_shuttle_paper.eps',
             format='eps',
             bbox_inches='tight')

plt.show()
