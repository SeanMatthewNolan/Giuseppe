import os
import pickle

import matplotlib.pyplot as plt
from numpy import arctan

os.chdir(os.path.dirname(__file__))  # Set directory to current location

# with open('guess.data', 'rb') as file:
#     sol = pickle.load(file)

with open('sol_set.data', 'rb') as file:
    sol_set = pickle.load(file)

sol = sol_set[-1]

max_u = sol.k[-1]
eps = sol.k[-2]

fig = plt.figure(figsize=(6.5, 5))
fig.suptitle('Predator Prey')

ax1 = fig.add_subplot(131)
ax1.set_xlabel('Time')
ax1.set_ylabel('Pestiside')
ax1.set_title('Pesticide (Control)')

ax2 = fig.add_subplot(132)
ax2.set_xlabel('Time')
ax2.set_ylabel('Crop')
ax2.set_title('Crop (Prey)')

ax3 = fig.add_subplot(133)
ax3.set_xlabel('Time')
ax3.set_ylabel('Insects')
ax3.set_title('Insects (Predator)')

for sol in sol_set[-2:-1]:
    ax1.plot(sol.t, max_u * arctan(sol.u[0, :] / eps) / 3.14159 + max_u / 2, linewidth=2)
    # ax1.plot(sol.t, sol.u[0, :], linewidth=2)
    ax2.plot(sol.t, sol.x[0, :], linewidth=2)
    ax3.plot(sol.t, sol.x[1, :], linewidth=2)

fig.tight_layout()

plt.show()
