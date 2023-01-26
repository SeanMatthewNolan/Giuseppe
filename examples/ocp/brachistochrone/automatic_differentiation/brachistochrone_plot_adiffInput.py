import pickle
import numpy as np
from brachistochrone_differentialAdiffInput import adiff_bvp
from casadi import cos, sin, tan, sqrt
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.formatter.useoffset'] = False

MED_FIG_SIZE = (6.5, 5)
SML_FIG_SIZE = (6.5, 3)

# Automatically Derived Dualized Necessary Conditions
dlam_ad = adiff_bvp.dual.ca_costate_dynamics
du_ad = adiff_bvp.control_handler.ca_control_dynamics
Hu_ad = adiff_bvp.dual.ca_dH_du
psi_adj_0_ad = adiff_bvp.dual.ca_adj_boundary_conditions.initial
psi_adj_f_ad = adiff_bvp.dual.ca_adj_boundary_conditions.terminal


# Analytically Derived Dualized Necessary Conditions
def dlam(t, x, lam, u, p, k):
    return np.array((0, 0, -lam[0] * cos(u[0]) - lam[1] * sin(u[0])))


def du(t, x, lam, u, p, k):
    return np.array((
        (k[0] * lam[0])
        / (x[2] * (lam[0] * cos(u[0]) + lam[1] * sin(u[0])) - lam[2] * k[0] * sin(u[0]))
    ))


def Hu(t, x, lam, u, p, k):
    return np.array((
        lam[1] * x[2] * cos(u[0]) - lam[0] * x[2] * sin(u[0]) - lam[2] * k[0] * cos(u[0])
    ))


def psi_adj_0(t, x, lam, u, p, nu0, k):
    return np.array((
        nu0[0] - lam[0] * x[2] * cos(u[0]) - lam[1] * x[2] * sin(u[0]) + lam[2] * k[0] * sin(u[0]),
        nu0[1] + lam[0],
        nu0[2] + lam[1],
        nu0[3] + lam[2],
    ))


def psi_adj_f(t, x, lam, u, p, nuf, k):
    return np.array((
        1 + lam[0] * x[2] * cos(u[0]) + lam[1] * x[2] * sin(u[0]) - lam[2] * k[0] * sin(u[0]),
        nuf[0] - lam[0],
        nuf[1] - lam[1],
        -lam[2],
    ))


def cot(x):
    return cos(x) / sin(x)


def csc(x):
    return 1 / sin(x)


def cycloid_constant2(t, x, lam, u, p, k):
    return cos(u[0]) ** 2 / (-2 * k[0] * x[1])


with open('../sol_set.data', 'rb') as file:
    sol = pickle.load(file)[-1]

err_dlam = []
err_du = []
err_Hu = []
err_psi_adj_0 = []
err_psi_adj_f = []
cycloid_constant2_list = []

for idx, t_val in enumerate(sol.t):
    err_dlam.append(
        dlam(t_val, sol.x[:, idx], sol.lam[:, idx], sol.u[:, idx], sol.p, sol.k)
        - dlam_ad(t_val, sol.x[:, idx], sol.lam[:, idx], sol.u[:, idx], sol.p, sol.k)
    )

    err_du.append(
        du(t_val, sol.x[:, idx], sol.lam[:, idx], sol.u[:, idx], sol.p, sol.k)
        - du_ad(t_val, sol.x[:, idx], sol.lam[:, idx], sol.u[:, idx], sol.p, sol.k)
    )

    err_Hu.append(
        Hu(t_val, sol.x[:, idx], sol.lam[:, idx], sol.u[:, idx], sol.p, sol.k)
        - Hu_ad(t_val, sol.x[:, idx], sol.lam[:, idx], sol.u[:, idx], sol.p, sol.k)
    )

    cycloid_constant2_list.append(cycloid_constant2(t_val, sol.x[:, idx], sol.lam[:, idx], sol.u[:, idx], sol.p, sol.k))

err_psi_adj_0 = psi_adj_0(t=sol.t[0], x=sol.x[:, 0], lam=sol.lam[:, 0], u=sol.u[:, 0], p=sol.p, nu0=sol.nu0, k=sol.k) \
    - psi_adj_0_ad(sol.t[0], sol.x[:, 0], sol.lam[:, 0], sol.u[:, 0], sol.p, sol.nu0, sol.k)

err_psi_adj_f = psi_adj_f(
    t=sol.t[-1], x=sol.x[:, -1], lam=sol.lam[:, -1], u=sol.u[:, -1], p=sol.p, nuf=sol.nuf, k=sol.k) \
    - psi_adj_f_ad(sol.t[-1], sol.x[:, -1], sol.lam[:, -1], sol.u[:, -1], sol.p, sol.nuf, sol.k)

cycloid_constant2_arr = np.array(cycloid_constant2_list)
idcs = np.where(np.logical_and(np.logical_not(np.isnan(cycloid_constant2_arr)), sol.t > 1e-3))
cycloid_constant2_arr = cycloid_constant2_arr[idcs]
t_cycloid = sol.t[idcs]

print('------- Analytical vs. AD Error -------')
print(f'Max err(dlam/dt)  = {np.max(err_dlam)}')
print(f'Max err(du/dt)    = {np.max(err_du)}')
print(f'Max err(Hu)       = {np.max(err_Hu)}')
print(f'Max err(Psi_adj0) = {np.max(err_psi_adj_0)}')
print(f'Max err(Psi_adjf) = {np.max(err_psi_adj_f)}')
print(f'Max diff(c)       = {np.max(cycloid_constant2_arr) - np.min(cycloid_constant2_arr)}')

# -----------------------------------------------------------
# PLOTTING
fig_cycloid = plt.figure(figsize=SML_FIG_SIZE)
ax1 = fig_cycloid.add_subplot(111)
ax1.grid()
ax1.plot(t_cycloid, cycloid_constant2_arr)
ax1.set_xlabel(r'$t$ [s]')
ax1.set_ylabel(r'$c^2$ [s$^3$/m$^2$]')

fig_cycloid.tight_layout()

fig_cycloid.savefig('brachistocrone_cycloid_constant.eps',
                    format='eps',
                    bbox_inches='tight')

plt.show()
