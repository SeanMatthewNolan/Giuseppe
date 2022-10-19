import pickle

import matplotlib.pyplot as plt
import numpy as np

with open('seed.data', 'rb') as file:
    sol = pickle.load(file)

# with open('sol_set.data', 'rb') as file:
#     sol_set = pickle.load(file)
# sol = sol_set[-1]

fig = plt.figure(figsize=(6.5, 5))
fig.suptitle('Orbit Raising')

ax1 = fig.add_subplot(211)
ax1.plot(sol.t, sol.u[0, :], linewidth=2)

ax2 = fig.add_subplot(312)
ax2.plot(sol.x[0, :] * np.sin(sol.x[1, :]), sol.x[0, :] * np.cos(sol.x[1, :]), linewidth=2)

ax3 = fig.add_subplot(313)
ax3.plot(sol.x[1, :], sol.x[0, :], linewidth=2)

fig.tight_layout()

plt.show()
