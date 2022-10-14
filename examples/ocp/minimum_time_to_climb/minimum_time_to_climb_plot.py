import pickle

import matplotlib.pyplot as plt
import numpy as np

from giuseppe.utils.examples import Atmosphere1976

from minimum_time_to_climb import S
from lookup_tables import thrust_table_bspline, eta_table_bspline_expanded, CLalpha_table_bspline_expanded,\
    CD0_table_bspline_expanded

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

r2d = 180 / np.pi

h = sol.x[0, :]
V = sol.x[1, :]
alpha = sol.u[0, :]
alpha_hat = alpha * r2d

atm = Atmosphere1976(use_metric=False)

T = np.asarray([atm.temperature(alt) for alt in h])
rho = np.asarray([atm.density(alt) for alt in h])
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
title = fig1.suptitle('Min. Time to Climb')

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

# FIGURE 2 LOOKUP TABLE VALUES
fig2 = plt.figure(figsize=MED_FIGSIZE)

ax21 = fig2.add_subplot(411)
ax21.plot(sol.t, thrust / 10_000)
ax21.grid()
ax21.set_ylabel('Thrust [10,000 lb]')

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
ax31.plot(sol.t, lift)
ax31.grid()
ax31.set_ylabel('Lift [lb]')

ax32 = fig3.add_subplot(312)
ax32.plot(sol.t, drag)
ax32.grid()
ax32.set_ylabel('Drag [lb]')

ax33 = fig3.add_subplot(313)
ax33.plot(sol.t, LoD)
ax33.grid()
ax33.set_ylabel('L/D')
ax33.set_xlabel(T_LAB)

fig3.tight_layout()

plt.show()
