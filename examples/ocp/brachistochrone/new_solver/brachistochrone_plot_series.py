import os
import pickle

import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))  # Set directory to file location

with open('sol_set.data', 'rb') as file:
    sol_set = pickle.load(file)

fig = plt.figure()
fig.suptitle('Brachistochrone')

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$\theta$')

fig_lam = plt.figure()
fig_lam.suptitle('Costates')

ax21 = fig_lam.add_subplot(3, 1, 1)
ax21.set_xlabel(r'$t$')
ax21.set_ylabel(r'$\lambda_x$')

ax22 = fig_lam.add_subplot(3, 1, 2)
ax22.set_xlabel(r'$t$')
ax22.set_ylabel(r'$\lambda_y$')

ax23 = fig_lam.add_subplot(3, 1, 3)
ax23.set_xlabel(r'$t$')
ax23.set_ylabel(r'$\lambda_v$')

for sol in sol_set:
    ax1.plot(sol.x[0, :], sol.x[1, :])
    ax2.plot(sol.t, sol.u[0, :])

    ax21.plot(sol.t, sol.lam[0, :])
    ax22.plot(sol.t, sol.lam[1, :])
    ax23.plot(sol.t, sol.lam[2, :])

fig_lam.tight_layout()

plt.show()
