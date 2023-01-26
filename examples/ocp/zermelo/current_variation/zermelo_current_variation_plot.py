import os;

os.chdir(os.path.dirname(__file__))  # Set diectory to current location

import pickle

import matplotlib.pyplot as plt

with open('../current_variation_sol_set.data', 'rb') as file:
    sol_set = pickle.load(file)

fig = plt.figure()
fig.suptitle('Zermelo Current Variation')

ax11 = fig.add_subplot(2, 1, 1)
ax11.set_xlabel(r'$x$')
ax11.set_ylabel(r'$y$')

ax12 = fig.add_subplot(2, 1, 2)
ax12.set_xlabel(r'$t$')
ax12.set_ylabel(r'$\theta$')

fig_lam = plt.figure()
fig_lam.suptitle('Costates')

ax21 = fig_lam.add_subplot(2, 1, 1)
ax21.set_xlabel(r'$t$')
ax21.set_ylabel(r'$\lambda_x$')

ax22 = fig_lam.add_subplot(2, 1, 2)
ax22.set_xlabel(r'$t$')
ax22.set_ylabel(r'$\lambda_y$')

for sol in sol_set:
    ax11.plot(sol.x[0, :], sol.x[1, :])
    ax12.plot(sol.t, sol.u[0, :])

    ax21.plot(sol.t, sol.lam[0, :])
    ax22.plot(sol.t, sol.lam[1, :])

fig_lam.tight_layout()

plt.show()
