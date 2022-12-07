import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

with open('bvp_sols.data', 'rb') as file:
    bvp_sols = pickle.load(file)

gradient = mpl.colormaps['viridis'].colors

continuation_idces = np.array((0, 1, 2, 3))

figs = []
figs_lam = []
for idx in continuation_idces:
    sol_set = bvp_sols[idx]
    sol = sol_set[-1]

    grad_idcs = np.int32(np.ceil(np.linspace(255, 0, len(sol_set))))


    def cols_gradient(n):
        return gradient[grad_idcs[n]]


    max_thrust = sol.k[0]
    h_ref = sol.k[4]
    eps = sol.k[-1]

    figs.append(plt.figure(figsize=(6.5, 5)))
    figs_lam.append(plt.figure(figsize=(6.5, 5)))
    fig = figs[-1]
    fig_lam = figs_lam[-1]

    # fig.suptitle(f'Continuation {idx}')
    # fig_lam.suptitle(f'Costate Continuation {idx}')

    ax1 = fig.add_subplot(211)
    ax1.grid()
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Thrust [lb]')
    # ax1.set_title('Thrust (Control)')

    ax21 = fig.add_subplot(234)
    ax21.grid()
    ax21.set_xlabel('Time [s]')
    ax21.set_ylabel('Altitude [ft]')
    # ax21.set_title('Altitude')

    ax22 = fig.add_subplot(235)
    ax22.grid()
    ax22.set_xlabel('Time [s]')
    ax22.set_ylabel('Velocity [ft/s]')
    # ax22.set_title('Velocity')

    ax23 = fig.add_subplot(236)
    ax23.grid()
    ax23.set_xlabel('Time [s]')
    ax23.set_ylabel('Mass [slug]')
    # ax23.set_title('Mass')

    ax_lam1 = fig_lam.add_subplot(311)
    ax_lam1.grid()
    ax_lam1.set_xlabel('Time [s]')
    ax_lam1.set_ylabel(r'$\lambda_h$ [1/s]')

    ax_lam2 = fig_lam.add_subplot(312)
    ax_lam2.grid()
    ax_lam2.set_xlabel('Time [s]')
    ax_lam2.set_ylabel(r'$\lambda_V$ [1/s$^2$]')

    ax_lam3 = fig_lam.add_subplot(313)
    ax_lam3.grid()
    ax_lam3.set_xlabel('Time [s]')
    ax_lam3.set_ylabel(r'$\lambda_m$ [ft-s/lb]')

    for idx_sol, sol in enumerate(sol_set):
        ax1.plot(sol.t, sol.u[0, :], linewidth=2, color=cols_gradient(idx_sol))
        # ax1.plot(sol.t, max_thrust/2 * np.sin(sol.u[0, :]) + max_thrust/2, linewidth=2, color=cols_gradient(idx_sol))
        ax21.plot(sol.t, sol.x[0, :], linewidth=2, color=cols_gradient(idx_sol))
        ax22.plot(sol.t, sol.x[1, :], linewidth=2, color=cols_gradient(idx_sol))
        ax23.plot(sol.t, sol.x[2, :], linewidth=2, color=cols_gradient(idx_sol))

        ax_lam1.plot(sol.t, sol.lam[0, :], linewidth=2, color=cols_gradient(idx_sol))
        ax_lam2.plot(sol.t, sol.lam[1, :], linewidth=2, color=cols_gradient(idx_sol))
        ax_lam3.plot(sol.t, sol.lam[2, :], linewidth=2, color=cols_gradient(idx_sol))

    fig.tight_layout()
    fig_lam.tight_layout()

figs[-1].savefig('goddard_BVP_iterations.eps',
                 format='eps',
                 bbox_inches='tight')

figs_lam[-1].savefig('goddard_BVP_costate_iterations.eps',
                     format='eps',
                     bbox_inches='tight')

plt.show()

