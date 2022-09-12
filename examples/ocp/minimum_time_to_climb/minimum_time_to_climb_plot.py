import pickle

import matplotlib.pyplot as plt
import numpy as np

with open('sol_set.data', 'rb') as file:
    sol = pickle.load(file)[-1]

t_lab = 'Time [sec]'
r2d = 180 / np.pi

fig = plt.figure(figsize=(6.5, 5))
title = fig.suptitle('Min. Time to Climb')

# Alt. vs. Time
ax1 = fig.add_subplot(321)
ax1.plot(sol.t, sol.x[0, :] / 1000)
xlabel_1 = ax1.set_xlabel(t_lab)
ylabel_1 = ax1.set_ylabel('Altitude [1000 ft]')
ax1.grid()

# Velocity vs. Time
ax2 = fig.add_subplot(322)
ax2.plot(sol.t, sol.x[1, :] / 100)
xlabel_2 = ax2.set_xlabel(t_lab)
ylabel_2 = ax2.set_ylabel('Velocity [100 ft/s]')
ax2.grid()

# FPA vs. Time
ax3 = fig.add_subplot(323)
ax3.plot(sol.t, sol.x[2, :] * r2d)
xlabel_3 = ax3.set_xlabel(t_lab)
ylabel_3 = ax3.set_ylabel(r'$\gamma$ [deg]')
ax3.grid()

# Weight vs. Time
ax4 = fig.add_subplot(324)
ax4.plot(sol.t, sol.x[3, :] / 10_000)
xlabel_4 = ax4.set_xlabel(t_lab)
ylabel_4 = ax4.set_ylabel('Weight [10,000 lb]')
ax4.grid()

# AoA vs. Time
ax5 = fig.add_subplot(325)
ax5.plot(sol.t, sol.u[0, :] * r2d)
xlabel_5 = ax5.set_xlabel(t_lab)
ylabel_5 = ax5.set_ylabel(r'$\alpha$ [deg]')
ax5.grid()

# Alt. Vs. Velocity
ax6 = fig.add_subplot(326)
ax6.plot(sol.x[1, :], sol.x[0, :])
xlabel_6 = ax6.set_xlabel('Velocity [100 ft/s]')
ylabel_6 = ax6.set_ylabel('Altitude [1000 ft]')
ax6.grid()

plt.show()
