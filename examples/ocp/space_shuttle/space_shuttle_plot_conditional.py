import pickle

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from giuseppe.utils.examples.atmosphere1976 import Atmosphere1976

from space_shuttle_footprint_conditional_atm import adiff_bvp


SMALL_FIGSIZE = (6.5, 3)
MED_FIG_SIZE = (6.5, 5)
BIG_FIG_SIZE = (6.5, 6.5)
T_LAB = r'$t$ [s]'

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
ax22.set_ylabel(r'$\beta$ [deg]')
ax22.set_xlabel(T_LAB)

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
ax32.set_ylabel(r'$-\dfrac{d\rho}{dh}$ [slug / 10$^9$ ft$^4$]')
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

ax4[4].legend()
ax4[-1].set_xlabel(T_LAB)
ax4[2].set_xlabel(T_LAB)

fig4.tight_layout()

# FIGURE 6 (VALIDATION)
ham_map = adiff_bvp.dual.ca_hamiltonian.map(len(sol_cond.t))
ham_u_func = adiff_bvp.dual.ca_dH_du
ham_uT_func = ca.Function('dH_du_T', ham_u_func.sx_in(), (ham_u_func(*ham_u_func.sx_in()).T,))
ham_u_map = ham_uT_func.map(len(sol_cond.t))
ham_t_map = adiff_bvp.dual.ca_dH_dt.map(len(sol_cond.t))

ham = np.asarray(ham_map(sol_cond.t, sol_cond.x, sol_cond.lam, sol_cond.u, sol_cond.p, sol_cond.k)).flatten()
ham_t_numerical = np.diff(ham) / np.diff(sol_cond.t)
ham_t_numerical_max = np.max(np.abs(ham_t_numerical))
ham_t = np.asarray(ham_t_map(sol_cond.t, sol_cond.x, sol_cond.lam, sol_cond.u, sol_cond.p, sol_cond.k)).flatten()
ham_t_max = np.max(np.abs(ham_t))
ham_u = np.asarray(ham_u_map(sol_cond.t, sol_cond.x, sol_cond.lam, sol_cond.u, sol_cond.p, sol_cond.k))
ham_alpha = ham_u[0, :]
ham_beta = ham_u[1, :]
ham_alpha_max = np.max(np.abs(ham_alpha))
ham_beta_max = np.max(np.abs(ham_beta))

psi_0 = np.asarray(adiff_bvp.ocp.ca_boundary_conditions.initial(
    sol_cond.t[0], sol_cond.x[:, 0], sol_cond.u[:, 0], sol_cond.p, sol_cond.k)).flatten()
psi_f = np.asarray(adiff_bvp.ocp.ca_boundary_conditions.terminal(
    sol_cond.t[-1], sol_cond.x[:, -1], sol_cond.u[:, -1], sol_cond.p, sol_cond.k)).flatten()
psi_adj_0 = np.asarray(adiff_bvp.dual.ca_adj_boundary_conditions.initial(
    sol_cond.t[0], sol_cond.x[:, 0], sol_cond.lam[:, 0], sol_cond.u[:, 0], sol_cond.p, sol_cond.nu0, sol_cond.k)).flatten()
psi_adj_f = np.asarray(adiff_bvp.dual.ca_adj_boundary_conditions.terminal(
    sol_cond.t[-1], sol_cond.x[:, -1], sol_cond.lam[:, -1], sol_cond.u[:, -1], sol_cond.p, sol_cond.nuf, sol_cond.k)).flatten()

fig6 = plt.figure(figsize=SMALL_FIGSIZE)
ax61 = fig6.add_subplot(211)
ax61.plot(sol_cond.t, ham_alpha * 1e9, label=r'$\alpha$')
ax61.plot(sol_cond.t, ham_beta * 1e9, zorder=0, label=r'$\beta$')
ax61.grid()
ax61.set_ylabel(r'$\partial H/\partial u$ [$10^{-9}$ 1/s]')
ax61.set_ylim((2*ax61.get_ylim()[0], ax61.get_ylim()[1]))
ax61.legend(loc='lower right')

ax62 = fig6.add_subplot(212)
ax62.plot(sol_cond.t, ham_t * r2d * 1e8, label='AD')
ax62.plot(sol_cond.t[:-1], ham_t_numerical * r2d * 1e8, zorder=0, label='FD')
ax62.grid()
ax62.set_ylabel(r'$\partial H/\partial t$ [$10^{-8}$ deg/s$^2$]')
ax62.set_xlabel(T_LAB)
ax62.legend()
ax62.set_ylim((ax62.get_ylim()[0], 2*ax62.get_ylim()[1]))
ax62.legend(loc='upper right')

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

fig4.savefig('space_shuttle_costates.eps',
             format='eps',
             bbox_inches='tight')

fig6.savefig('space_shuttle_hamiltonian.eps',
             format='eps',
             bbox_inches='tight')

plt.show()
