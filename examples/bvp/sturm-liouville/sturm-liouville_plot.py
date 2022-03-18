import pickle

import matplotlib.pyplot as plt

with open('sol.data', 'rb') as file:
    sol = pickle.load(file)

fig = plt.figure()
fig.suptitle('Sturm-Liouville')

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(sol.t, sol.x[0, :])
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(sol.t, sol.x[1, :])
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y^\prime$')

fig.tight_layout()

plt.show()
