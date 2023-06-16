import pickle

import matplotlib.pyplot as plt
import numpy as np

from giuseppe.utils.examples import Atmosphere1976

# from minimum_time_to_climb import S, adiff_dual
from lookup_tables import thrust_table_bspline, eta_table_bspline_expanded, CLalpha_table_bspline_expanded,\
    CD0_table_bspline_expanded, temp_table_bspline, dens_table_bspline

SMALL_FIGSIZE = (6.5, 3)
MED_FIGSIZE = (6.5, 5)
LARGE_FIGSIZE = (6.5, 7.5)
T_LAB = 'Time [sec]'

DATA = 0

if DATA == 0:
    with open('sol_set.data', 'rb') as file:
        sols = pickle.load(file)
        sol = sols[-1]
elif DATA == 1:
    with open('seed.data', 'rb') as file:
        sol = pickle.load(file)
elif DATA == 2:
    with open('guess.data', 'rb') as file:
        sol = pickle.load(file)

# noinspection PyTypeChecker
np.savetxt('txuSolution.csv', np.vstack((sol.t.reshape((1, -1)), sol.x, sol.u)), delimiter=',')

r2d = 180 / np.pi
d2r = np.pi / 180

S = 530
h = sol.x[0, :]
V = sol.x[1, :]
alpha = sol.u[0, :]
alpha_hat = alpha * r2d

atm = Atmosphere1976(use_metric=False)

# T = np.asarray([atm.temperature(alt) for alt in h])
# rho = np.asarray([atm.density(alt) for alt in h])
T = np.asarray(temp_table_bspline(h)).flatten()
rho = np.asarray(dens_table_bspline(h)).flatten()

a = np.sqrt(atm.specific_heat_ratio * atm.gas_constant * T)

M = V/a
Qdyn = 0.5 * rho * V**2

thrust = np.asarray(thrust_table_bspline(np.vstack((M.T, h.T)))).flatten()
eta = np.asarray(eta_table_bspline_expanded(M)).flatten()
CLalpha = np.asarray(CLalpha_table_bspline_expanded(M)).flatten()
CD0 = np.asarray(CD0_table_bspline_expanded(M)).flatten()

CD = CD0 + eta * CLalpha * alpha_hat**2
CL = CLalpha * alpha_hat

LoD = CL / CD

drag = 0.5 * CD * S * rho * V**2
lift = 0.5 * CL * S * rho * V**2

# FIGURE 1 (STATES)
fig1 = plt.figure(figsize=MED_FIGSIZE)
# title = fig1.suptitle('Min. Time to Climb')

# Alt. vs. Time
ax1 = fig1.add_subplot(321)
ax1.plot(sol.t, sol.x[0, :] / 1_000)
xlabel_1 = ax1.set_xlabel(T_LAB)
ylabel_1 = ax1.set_ylabel('Altitude [1000 ft]')
ax1.grid()

# Velocity vs. Time
ax2 = fig1.add_subplot(322)
ax2.plot(sol.t, sol.x[1, :] / 100)
xlabel_2 = ax2.set_xlabel(T_LAB)
ylabel_2 = ax2.set_ylabel('Velocity [100 ft/s]')
ax2.grid()

# FPA vs. Time
ax3 = fig1.add_subplot(323)
ax3.plot(sol.t, sol.x[2, :] * r2d)
xlabel_3 = ax3.set_xlabel(T_LAB)
ylabel_3 = ax3.set_ylabel(r'$\gamma$ [deg]')
ax3.grid()

# Weight vs. Time
ax4 = fig1.add_subplot(324)
ax4.plot(sol.t, sol.x[3, :] / 10_000)
xlabel_4 = ax4.set_xlabel(T_LAB)
ylabel_4 = ax4.set_ylabel('Weight [10,000 lb]')
ax4.grid()

# AoA vs. Time
ax5 = fig1.add_subplot(325)
ax5.plot(sol.t, sol.u[0, :] * r2d)
xlabel_5 = ax5.set_xlabel(T_LAB)
ylabel_5 = ax5.set_ylabel(r'$\alpha$ [deg]')
ax5.grid()

# Alt. Vs. Velocity
ax6 = fig1.add_subplot(326)
ax6.plot(sol.x[1, :] / 100, sol.x[0, :] / 1_000)
xlabel_6 = ax6.set_xlabel('Velocity [100 ft/s]')
ylabel_6 = ax6.set_ylabel('Altitude [1000 ft]')
ax6.grid()

fig1.tight_layout()

# FIGURE 2 AERO COEFFICIENTS
fig2 = plt.figure(figsize=MED_FIGSIZE)

ax21 = fig2.add_subplot(411)
ax21.plot(sol.t, abs(LoD))
ax21.grid()
ax21.set_ylabel('L/D')
# ax21.set_xlabel(T_LAB)

ax22 = fig2.add_subplot(412)
ax22.plot(sol.t, CLalpha)
ax22.grid()
ax22.set_ylabel(r'$C_{L,\alpha}$')

ax23 = fig2.add_subplot(413)
ax23.plot(sol.t, CD0)
ax23.grid()
ax23.set_ylabel(r'$C_{D,0}$')

ax24 = fig2.add_subplot(414)
ax24.plot(sol.t, eta)
ax24.grid()
ax24.set_ylabel(r'$\eta$')
ax24.set_xlabel(T_LAB)

fig2.tight_layout()

# FIGURE 3 AERO FORCES
fig3 = plt.figure(figsize=MED_FIGSIZE)

ax31 = fig3.add_subplot(311)
ax31.plot(sol.t, lift / 1_000_000)
ax31.grid()
ax31.set_ylabel('Lift [1,000,000 lb]')

ax32 = fig3.add_subplot(312)
ax32.plot(sol.t, drag / 1_000_000)
ax32.grid()
ax32.set_ylabel('Drag [1,000,000 lb]')

ax33 = fig3.add_subplot(313)
ax33.plot(sol.t, thrust / 10_000)
ax33.grid()
ax33.set_ylabel('Thrust [10,000 lb]')
ax33.set_xlabel(T_LAB)

fig3.tight_layout()

# FIGURE 4 (ATMOSPHERE)
fig4 = plt.figure(figsize=MED_FIGSIZE)
ax41 = fig4.add_subplot(311)
ax41.plot(sol.t, M)
ax41.grid()
ax41.set_ylabel('Mach')

ax42 = fig4.add_subplot(312)
ax42.plot(sol.t, a)
ax42.grid()
ax42.set_ylabel(r'$a$ [ft/s]')

ax43 = fig4.add_subplot(313)
ax43.plot(sol.t, Qdyn)
ax43.grid()
ax43.set_ylabel(r'$Q_\infty$ [psf]')
ax43.set_xlabel(T_LAB)

# FIGURE 5 (COSTATES)
fig5 = plt.figure(figsize=SMALL_FIGSIZE)

# Alt. vs. Time
ax51 = fig5.add_subplot(221)
ax51.plot(sol.t, sol.lam[0, :])
# ax51.set_xlabel(T_LAB)
ax51.set_ylabel(r'$\lambda_h$ [s/ft]')
ax51.grid()

# Velocity vs. Time
ax52 = fig5.add_subplot(222)
ax52.plot(sol.t, sol.lam[1, :])
# ax52.set_xlabel(T_LAB)
ax52.set_ylabel(r'$\lambda_V$ [s$^2$/ft]')
ax52.grid()

# FPA vs. Time
ax53 = fig5.add_subplot(223)
ax53.plot(sol.t, sol.lam[2, :] * d2r)
ax53.set_xlabel(T_LAB)
ax53.set_ylabel(r'$\lambda_\gamma$ [s/deg]')
ax53.grid()

# Weight vs. Time
ax54 = fig5.add_subplot(224)
ax54.plot(sol.t, sol.lam[3, :])
ax54.set_xlabel(T_LAB)
ax54.set_ylabel(r'$\lambda_W$ [s/lb]')
ax54.grid()

fig5.tight_layout()

# # FIGURE 6 (Validation with Hamiltonian)
# ham_map = adiff_dual.ca_hamiltonian.map(len(sol.t))
# ham_u_map = adiff_dual.ca_dh_du.map(len(sol.t))
# ham_t_map = adiff_dual.ca_dh_dt.map(len(sol.t))
#
# ham = np.asarray(ham_map(sol.t, sol.x, sol.lam, sol.u, sol.p, sol.k)).flatten()
# ham_t_numerical = np.diff(ham) / np.diff(sol.t)
# ham_t_numerical_max = np.max(np.abs(ham_t_numerical))
# ham_u = np.asarray(ham_u_map(sol.t, sol.x, sol.lam, sol.u, sol.p, sol.k)).flatten()
# ham_u_max = np.max(np.abs(ham_u))
# ham_t = np.asarray(ham_t_map(sol.t, sol.x, sol.lam, sol.u, sol.p, sol.k)).flatten()
# ham_t_max = np.max(np.abs(ham_t))
#
# psi_0 = np.asarray(adiff_dual.ca_initial_boundary_conditions(
#     sol.t[0], sol.x[:, 0], sol.u[:, 0], sol.p, sol.k)).flatten()
# psi_f = np.asarray(adiff_dual.ca_terminal_boundary_conditions(
#     sol.t[-1], sol.x[:, -1], sol.u[:, -1], sol.p, sol.k)).flatten()
# psi_adj_0 = np.asarray(adiff_dual.ca_initial_adjoint_boundary_conditions(
#     sol.t[0], sol.x[:, 0], sol.lam[:, 0], sol.u[:, 0], sol.p, sol.nu0, sol.k)).flatten()
# psi_adj_f = np.asarray(adiff_dual.ca_terminal_adjoint_boundary_conditions(
#     sol.t[-1], sol.x[:, -1], sol.lam[:, -1], sol.u[:, -1], sol.p, sol.nuf, sol.k)).flatten()
#
# fig6 = plt.figure(figsize=SMALL_FIGSIZE)
# ax61 = fig6.add_subplot(211)
# ax61.plot(sol.t, ham_u * d2r * 1e6)
# ax61.grid()
# ax61.set_ylabel(r'$\partial H/\partial u$ [$10^{-6}$ 1/deg]')
#
# ax62 = fig6.add_subplot(212)
# ax62.plot(sol.t, ham_t * 1e5, label='AD')
# ax62.plot(sol.t[:-1], ham_t_numerical * 1e5, zorder=0, label='FD')
# ax62.grid()
# ax62.set_ylabel(r'$\partial H/\partial t$ [$10^{-5}$ 1/s]')
# ax62.set_xlabel(T_LAB)
# ax62.set_ylim((1.5*ax62.get_ylim()[0], -1.5*ax62.get_ylim()[0]))
# ax62.legend(loc='upper center')
#
# fig6.tight_layout()

# print(f'Max dH/du = {ham_u_max * d2r:.4} [1/deg]')
# print(f'Max dH/dt (AD) = {ham_t_max:.4} [1/s]')
# print(f'Max dH/dt (Num.) = {ham_t_numerical_max:.4} [1/s]')

# print(f'\nt - t0 = {psi_0[0]:.4} [s]')
# print(f'h - h0 = {psi_0[1]:.4} [ft]')
# print(f'V - V0 = {psi_0[2]:.4} [ft/s]')
# print(f'gam - gam0 = {psi_0[3] * r2d:.4} [deg]')
# print(f'W - W0 = {psi_0[4]:.4} [lb]')
# print(f'h - hf = {psi_f[0]:.4} [ft]')
# print(f'V - Vf = {psi_f[1]:.4} [ft/s]')
# print(f'gam - gamf = {psi_f[2] * r2d:.4} [deg]')

# print(f'\nPhi0_adj_t - H0 = {psi_adj_0[0]:.4} [1]')
# print(f'Phi0_adj_h + lam0_h = {psi_adj_0[1]:.4} [1/ft]')
# print(f'Phi0_adj_V + lam0_V = {psi_adj_0[2]:.4} [s/ft]')
# print(f'Phi0_adj_gam + lam0_gam = {psi_adj_0[3] * d2r:.4} [1/deg]')
# print(f'Phi0_adj_W + lam0_W = {psi_adj_0[4]:.4} [1/lb]')

# print(f'\nPhif_adj_t + Hf = {psi_adj_f[0]:.4} [1]')
# print(f'Phif_adj_h + lamf_h = {psi_adj_f[1]:.4} [1/ft]')
# print(f'Phif_adj_V + lamf_V = {psi_adj_f[2]:.4} [s/ft]')
# print(f'Phif_adj_gam + lamf_gam = {psi_adj_f[3] * d2r:.4} [1/deg]')
# print(f'Phif_adj_W + lamf_W = {psi_adj_f[4]:.4} [1/lb]')

# SAVE FIGURES
fig1.savefig('mttc_states.eps',
             format='eps',
             bbox_inches='tight')

fig2.savefig('mttc_aero_coeffs.eps',
             format='eps',
             bbox_inches='tight')

fig3.savefig('mttc_forces.eps',
             format='eps',
             bbox_inches='tight')

fig4.savefig('mttc_atmosphere.eps',
             format='eps',
             bbox_inches='tight')

fig5.savefig('mttc_costates.eps',
             format='eps',
             bbox_inches='tight')

fig5.savefig('mttc_costates.eps',
             format='eps',
             bbox_inches='tight')

# fig6.savefig('mttc_hamiltonian.eps',
#              format='eps',
#              bbox_inches='tight')

plt.show()
