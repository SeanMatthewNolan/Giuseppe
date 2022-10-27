import pickle

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from giuseppe.utils.examples.atmosphere1976 import Atmosphere1976

from space_shuttle_footprint_adiffInput import adiff_bvp


SMALL_FIGSIZE = (6.5, 3)
MED_FIG_SIZE = (6.5, 5)
BIG_FIG_SIZE = (6.5, 6.5)
T_LAB = 'Time [s]'

r2d = 180 / np.pi
d2r = np.pi / 180

with open('sol_set.data', 'rb') as file:
    sol = pickle.load(file)[-1]
with open('sol_set_25_000.data', 'rb') as file:
    sol_25k = pickle.load(file)[-1]
with open('sol_set_conditional.data', 'rb') as file:
    sol_cond = pickle.load(file)[-1]
with open('sol_set_conditional_25_000.data', 'rb') as file:
    sol_cond_25k = pickle.load(file)[-1]

# FIGURE 1 (STATES)
fig1 = plt.figure(figsize=BIG_FIG_SIZE)

plot_nums = (1, 3, 5, 2, 4, 6)
titles = ('Altitude [100,000 ft]', 'Cross-Range [deg]', 'Down-Range [deg]',
          'Velocity [1,000 ft/s]', 'Flight Path Angle [deg]', 'Heading Angle [deg]')
mult = (1e-5, r2d, r2d, 1e-3, r2d, r2d)
ax1 = []
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for idx, plot_num in enumerate(plot_nums):
    ax1.append(fig1.add_subplot(3, 2, plot_num))
    h1 = ax1[idx].plot(sol.t, sol.x[idx, :] * mult[idx], color=colors[0], label=r'Exp. ($h_f = 80,000$ ft)')
    h2 = ax1[idx].plot(sol_cond.t, sol_cond.x[idx, :] * mult[idx], color=colors[1], label=r'Cond. ($h_f = 80,000$ ft)')
    h3 = ax1[idx].plot(sol_25k.t, sol_25k.x[idx, :] * mult[idx], '--', color=colors[0],
                       label=r'Exp. ($h_f = 25,000$ ft)')
    h4 = ax1[idx].plot(sol_cond_25k.t, sol_cond_25k.x[idx, :] * mult[idx], '--', color=colors[1],
                       label=r'Cond. ($h_f = 25,000$ ft)')

    h1[0].set_zorder(1)
    h2[0].set_zorder(0)
    h3[0].set_zorder(3)
    h4[0].set_zorder(2)

    ax1[idx].set_xlabel(T_LAB)
    ax1[idx].set_title(titles[idx])
    ax1[idx].grid()

ax1[4].legend()

fig1.tight_layout()

# FIGURE 2 (CONTROL HISTORIES)
fig2 = plt.figure(figsize=MED_FIG_SIZE)

plot_nums = (1, 3, 2, 4)
indices = (0, 1, 0, 1)
mult = (r2d, r2d, r2d, r2d)
line_styles = ('-', '--')
ax2 = []

for ax_idx, plot_num in enumerate(plot_nums):
    ax2.append(fig2.add_subplot(2, 2, plot_num))
    idx = indices[ax_idx]

    if plot_num % 2 == 1:  # Left Column of plots
        ax2[ax_idx].plot(sol.t, sol.u[idx, :] * mult[idx], line_styles[idx], label='Exp.')
        ax2[ax_idx].plot(sol_cond.t, sol_cond.u[idx, :] * mult[idx], zorder=0, label='Cond.')
        ax2[ax_idx].grid()
    else:  # Right column of plots
        ax2[ax_idx].plot(sol_25k.t, sol_25k.u[idx, :] * mult[idx], label='Exp.')
        ax2[ax_idx].plot(sol_cond_25k.t, sol_cond_25k.u[idx, :] * mult[idx], zorder=0, label='Cond.')
        ax2[ax_idx].grid()

ax2[0].set_ylabel('Angle of Attack [deg]')
ax2[0].set_title(r'$h_f = 80,000$ ft')
ax2[1].set_ylabel('Bank Angle [deg]')
ax2[1].set_xlabel(T_LAB)
ax2[2].set_title(r'$h_f = 25,000$ ft')
ax2[3].set_xlabel(T_LAB)

fig2.tight_layout()

# FIGURE 3 (ATMOSPHERE)
re = 20_902_900
mu = 0.14076539e17
g0 = mu / re**2
atm = Atmosphere1976(use_metric=False, earth_radius=re, gravity=g0)

rho_0 = 0.002378  # slug/ft^3
h_ref = 23_800  # ft

dens_exp = rho_0 * np.exp(-sol.x[0, :] / h_ref)
dens_exp_deriv = -rho_0 / h_ref * np.exp(-sol.x[0, :] / h_ref)
dens_exp_25k = rho_0 * np.exp(-sol_25k.x[0, :] / h_ref)
dens_exp_deriv_25k = -rho_0 / h_ref * np.exp(-sol_25k.x[0, :] / h_ref)

h_sx = ca.SX.sym('h', 1)
_, __, dens_expr = atm.get_ca_atm_expr(h_sx)
dens_deriv_expr = ca.jacobian(dens_expr, h_sx)

dens_ca_func = ca.Function('rho', (h_sx,), (dens_expr,), ('h',), ('rho',))
dens_deriv_ca_func = ca.Function('drho_dh', (h_sx,), (dens_deriv_expr,), ('h',), ('drho_dh',))

dens_cond = np.empty(shape=sol_cond.x[0, :].shape)
dens_cond_deriv = np.empty(shape=sol_cond.x[0, :].shape)
dens_cond_25k = np.empty(shape=sol_cond_25k.x[0, :].shape)
dens_cond_deriv_25k = np.empty(shape=sol_cond_25k.x[0, :].shape)
layer_cond = list()
layer_cond_25k = list()

for i, h in enumerate(sol_cond.x[0, :]):
    dens_cond[i] = dens_ca_func(h)
    dens_cond_deriv[i] = dens_deriv_ca_func(h)
    layer_cond.append(atm.layer(h))

for i, h in enumerate(sol_cond_25k.x[0, :]):
    dens_cond_25k[i] = dens_ca_func(h)
    dens_cond_deriv_25k[i] = dens_deriv_ca_func(h)
    layer_cond_25k.append(atm.layer(h))

layer_cond = np.array(layer_cond)
layer_cond_25k = np.array(layer_cond_25k)

fig3 = plt.figure(figsize=MED_FIG_SIZE)

ax31 = fig3.add_subplot(221)
ax32 = fig3.add_subplot(222)
ax33 = fig3.add_subplot(223)
ax34 = fig3.add_subplot(224)

ax31.plot(sol.t, dens_exp * 100_000, label='Exponential Atm.')
ax31.plot(sol_cond.t, dens_cond * 100_000,  zorder=0, label='Conditional Atm.')

ax32.plot(sol_25k.t, dens_exp_25k * 100_000, label='Exponential')
ax32.plot(sol_cond_25k.t, dens_cond_25k * 100_000,  zorder=0, label='Conditional Atm.')

ax33.plot(sol.t, dens_exp_deriv * 1e9, label='Exponential')
ax33.plot(sol_cond.t, dens_cond_deriv * 1e9,  zorder=0, label='Conditional Atm.')

ax34.plot(sol_25k.t, dens_exp_deriv_25k * 1e9, label='Exponential')
ax34.plot(sol_cond_25k.t, dens_cond_deriv_25k * 1e9, zorder=0, label='Conditional Atm.')

# for layer in atm.layer_names:
#     layer_idcs = np.where(layer_cond == layer)
#     if len(layer_idcs[0]) > 0:
#         ax31.plot(sol_cond.t[layer_idcs], dens_cond[layer_idcs] * 100_000, label=layer)
#         ax33.plot(sol_cond.t[layer_idcs], dens_cond_deriv[layer_idcs] * 1e9, label=layer)
#
#     layer_idcs = np.where(layer_cond_25k == layer)
#     if len(layer_idcs[0]) > 0:
#         ax32.plot(sol_cond_25k.t[layer_idcs], dens_cond_25k[layer_idcs] * 100_000, label=layer)
#         ax34.plot(sol_cond_25k.t[layer_idcs], dens_cond_deriv_25k[layer_idcs] * 100_000, label=layer)

ax31.grid()
ax32.grid()
ax33.grid()
ax34.grid()

ax33.set_xlabel(T_LAB)
ax34.set_xlabel(T_LAB)
ax31.set_ylabel(r'$\rho$ [slug / 100,000 ft$^3$]')
ax33.set_ylabel(r'$\dfrac{d\rho}{dh}$ [slug / 10$^9$ ft$^4$]')
ax31.set_title(r'$h_f = 80,000$ ft')
ax32.set_title(r'$h_f = 25,000$ ft')

ax32.legend()

fig3.tight_layout()

# FIGURE 4 (VALIDATION)
ham_map = adiff_bvp.dual.ca_hamiltonian.map(len(sol.t))
ham_u_func = adiff_bvp.dual.ca_dH_du
ham_uT_func = ca.Function('dH_du_T', ham_u_func.sx_in(), (ham_u_func(*ham_u_func.sx_in()).T,))
ham_u_map = ham_uT_func.map(len(sol.t))
ham_t_map = adiff_bvp.dual.ca_dH_dt.map(len(sol.t))

ham = np.asarray(ham_map(sol.t, sol.x, sol.lam, sol.u, sol.p, sol.k)).flatten()
ham_t_numerical = np.diff(ham) / np.diff(sol.t)
ham_t_numerical_max = np.max(np.abs(ham_t_numerical))
ham_t = np.asarray(ham_t_map(sol.t, sol.x, sol.lam, sol.u, sol.p, sol.k)).flatten()
ham_t_max = np.max(np.abs(ham_t))
ham_u = np.asarray(ham_u_map(sol.t, sol.x, sol.lam, sol.u, sol.p, sol.k))
ham_alpha = ham_u[0, :]
ham_beta = ham_u[1, :]
ham_alpha_max = np.max(np.abs(ham_alpha))
ham_beta_max = np.max(np.abs(ham_beta))

psi_0 = np.asarray(adiff_bvp.ocp.ca_boundary_conditions.initial(
    sol.t[0], sol.x[:, 0], sol.u[:, 0], sol.p, sol.k)).flatten()
psi_f = np.asarray(adiff_bvp.ocp.ca_boundary_conditions.terminal(
    sol.t[-1], sol.x[:, -1], sol.u[:, -1], sol.p, sol.k)).flatten()
psi_adj_0 = np.asarray(adiff_bvp.dual.ca_adj_boundary_conditions.initial(
    sol.t[0], sol.x[:, 0], sol.lam[:, 0], sol.u[:, 0], sol.p, sol.nu0, sol.k)).flatten()
psi_adj_f = np.asarray(adiff_bvp.dual.ca_adj_boundary_conditions.terminal(
    sol.t[-1], sol.x[:, -1], sol.lam[:, -1], sol.u[:, -1], sol.p, sol.nuf, sol.k)).flatten()

fig6 = plt.figure(figsize=SMALL_FIGSIZE)
ax61 = fig6.add_subplot(211)
ax61.plot(sol.t, ham_alpha * 1e9, label=r'$\alpha$')
ax61.plot(sol.t, ham_beta * 1e9, zorder=0, label=r'$\beta$')
ax61.grid()
ax61.set_ylabel(r'$\partial H/\partial u$ [$10^{-9}$ 1/s]')
ax61.set_ylim((2*ax61.get_ylim()[0], ax61.get_ylim()[1]))
ax61.legend(loc='lower center')

ax62 = fig6.add_subplot(212)
ax62.plot(sol.t, ham_t * r2d, label='AD')
ax62.plot(sol.t[:-1], ham_t_numerical * r2d * 1e10, zorder=0, label='FD')
ax62.grid()
ax62.set_ylabel(r'$\partial H/\partial t$ [$10^{-10}$ deg/s$^2$]')
ax62.set_xlabel(T_LAB)
ax62.legend()
ax62.set_ylim((2.5*ax62.get_ylim()[0], ax62.get_ylim()[1]))
ax62.legend(loc='lower center')

fig6.tight_layout()

print(f'Max dH/dalpha = {ham_alpha_max:.4} [1/s]')
print(f'Max dH/dbeta = {ham_beta_max:.4} [1/s]')
print(f'Max dH/dt (AD) = {ham_t_max * r2d:.4} [deg/s2]')
print(f'Max dH/dt (Num.) = {ham_t_numerical_max * r2d:.4} [deg/s2]')

print(f'\nt - t0 = {psi_0[0]:.4} [s]')
print(f'h - h0 = {psi_0[1]:.4} [ft]')
print(f'phi - phi0 = {psi_0[2] * r2d:.4} [deg]')
print(f'tha - tha0 = {psi_0[3] * r2d:.4} [deg]')
print(f'V - V0 = {psi_0[4]:.4} [ft/s]')
print(f'gam - gam0 = {psi_0[5] * r2d:.4} [deg]')
print(f'psi - psi0 = {psi_0[6] * r2d:.4} [deg]')

print(f'\nh - hf = {psi_f[0]:.4} [ft]')
print(f'V - Vf = {psi_f[1]:.4} [ft/s]')
print(f'gam - gamf = {psi_f[2] * r2d:.4} [deg]')

print(f'\nPhi0_adj_t - H0 = {psi_adj_0[0] * r2d:.4} [deg/s]')
print(f'Phi0_adj_h + lam0_h = {psi_adj_0[1] * r2d:.4} [deg/ft-s]')
print(f'Phi0_adj_phi + lam0_phi = {psi_adj_0[2]:.4} [1/s]')
print(f'Phi0_adj_tha + lam0_tha = {psi_adj_0[3]:.4} [1/s]')
print(f'Phi0_adj_V + lam0_V = {psi_adj_0[4] * r2d:.4} [deg/ft]')
print(f'Phi0_adj_gam + lam0_gam = {psi_adj_0[5]:.4} [1/s]')
print(f'Phi0_adj_psi + lam0_psi = {psi_adj_0[6]:.4} [1/s]')

print(f'\nPhif_adj_t + Hf = {psi_adj_f[0] * r2d:.4} [deg/s]')
print(f'Phif_adj_h + lamf_h = {psi_adj_f[1] * r2d:.4} [deg/ft-s]')
print(f'Phif_adj_phi + lamf_phi = {psi_adj_f[2]:.4} [1/s]')
print(f'Phif_adj_tha + lamf_tha = {psi_adj_f[3]:.4} [1/s]')
print(f'Phif_adj_V + lamf_V = {psi_adj_f[4] * r2d:.4} [deg/ft]')
print(f'Phif_adj_gam + lamf_gam = {psi_adj_f[5]:.4} [1/s]')
print(f'Phif_adj_psi + lamf_psi = {psi_adj_f[6]:.4} [1/lb]')

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

fig6.savefig('space_shuttle_hamiltonian.eps',
             format='eps',
             bbox_inches='tight')

plt.show()
