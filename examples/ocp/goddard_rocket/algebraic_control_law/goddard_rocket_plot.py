import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(__file__))  # Set directory to current location

with open('sol_set.data', 'rb') as file:
    sol_set = pickle.load(file)

sol = sol_set[-1]

max_thrust = sol.k[0]
h_ref = sol.k[4]
eps = sol.k[-1]

fig = plt.figure(figsize=(6.5, 5))
fig.suptitle('Goddard Rocket')

ax1 = fig.add_subplot(211)
ax1.plot(sol.t, max_thrust * np.arctan(sol.u[0, :] / (eps * h_ref)) / np.pi + max_thrust / 2, linewidth=2)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Thrust [lb]')
ax1.set_title('Thrust (Control)')

ax21 = fig.add_subplot(234)
ax21.plot(sol.t, sol.x[0, :], linewidth=2)
ax21.set_xlabel('Time [s]')
ax21.set_ylabel('Altitude [ft]')
ax21.set_title('Altitude')

ax22 = fig.add_subplot(235)
ax22.plot(sol.t, sol.x[1, :], linewidth=2)
ax22.set_xlabel('Time [s]')
ax22.set_ylabel('Velocity [ft/s]')
ax22.set_title('Velocity')

ax22 = fig.add_subplot(236)
ax22.plot(sol.t, sol.x[2, :], linewidth=2)
ax22.set_xlabel('Time [s]')
ax22.set_ylabel('Mass [slug]')
ax22.set_title('Mass')

fig.tight_layout()

plt.show()
