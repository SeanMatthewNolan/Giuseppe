import os

import numpy as np
import pickle

import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))  # Set directory to current location

with open('sol_set.data', 'rb') as file:
    sol_set = pickle.load(file)

sol = sol_set[-1]

g0 = 1.0  # Gravity at surface [-]
h0 = 1.0  # Initial height
v0 = 0.0  # Initial velocity
m0 = 1.0  # Initial mass
Tc = 3.5  # Use for thrust
Hc = 500  # Use for drag
Vc = 620  # Use for drag
Mc = 0.6  # Fraction of initial mass left at end
c = 0.5 * np.sqrt(g0 * h0)  # Thrust-to-fuel mass
mf = Mc * m0  # Final mass
Dc = 0.5 * Vc * m0 / g0  # Drag scaling
T_max = Tc * g0 * m0  # Maximum thrust

u_max = T_max
u_min = 0
u = 0.5 * ((u_max - u_min) * np.sin(sol.u) + u_max + u_min)

fig = plt.figure(figsize=(6.5, 5))
fig.suptitle('Goddard Rocket (Nondimensionalized)')

ax1 = fig.add_subplot(211)
ax1.plot(sol.t, u[0, :], linewidth=2)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Thrust [1/s]')
ax1.set_title('Thrust (Control)')

ax21 = fig.add_subplot(234)
ax21.plot(sol.t, sol.x[0, :], linewidth=2)
ax21.set_xlabel('Time [s]')
ax21.set_ylabel('Altitude [-]')
ax21.set_title('Altitude')

ax22 = fig.add_subplot(235)
ax22.plot(sol.t, sol.x[1, :], linewidth=2)
ax22.set_xlabel('Time [s]')
ax22.set_ylabel('Velocity [-/s]')
ax22.set_title('Velocity')

ax22 = fig.add_subplot(236)
ax22.plot(sol.t, sol.x[2, :], linewidth=2)
ax22.set_xlabel('Time [s]')
ax22.set_ylabel('Mass [-]')
ax22.set_title('Mass')

fig.tight_layout()

plt.show()