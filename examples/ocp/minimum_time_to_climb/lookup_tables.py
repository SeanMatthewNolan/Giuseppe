import casadi as ca
import numpy as np
from scipy.interpolate import PchipInterpolator, interp2d, bisplrep, bisplev

from giuseppe.utils.examples.atmosphere1976 import Atmosphere1976

h = ca.MX.sym('h', 1)
M = ca.MX.sym('M', 1)
interp_method = 'bspline'  # either 'bspline' or 'linear'

M_grid_thrust = np.array((0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8))
h_grid_thrust = np.array((-2, 0, 5, 10, 15, 20, 25, 30, 40, 50, 70)) * 1e3

data_thrust_original = np.array(((np.nan, 24.2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
                                 (np.nan, 28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, np.nan, np.nan, np.nan),
                                 (np.nan, 28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, np.nan),
                                 (np.nan, 30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, np.nan),
                                 (np.nan, 34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1),
                                 (np.nan, 37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4),
                                 (np.nan, 36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7),
                                 (np.nan, np.nan, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0, 2.2),
                                 (np.nan, np.nan, np.nan, np.nan, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9),
                                 (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 34.6, 31.1, 21.7, 13.3, 3.1))) * 1e3


def extrapolate(x, y, z, x_idx_new, y_idx_new, x_idx_old, y_idx_old, z_min=0.0):
    z_old = z[x_idx_old, y_idx_old]

    if x_idx_new == x_idx_old:
        # Extrapolate along y only
        idx_dir = np.sign(y_idx_new - y_idx_old)
        dz_dy = (z_old - z[x_idx_old, y_idx_old - idx_dir]) / (y[y_idx_old] - y[y_idx_old - idx_dir])
        dy = y[y_idx_new] - y[y_idx_old]
        z_new = max(float(z_old + dz_dy * dy), z_min)
    elif y_idx_new == y_idx_old:
        # Extrapolate along x only
        idx_dir = np.sign(x_idx_new - x_idx_old)
        dz_dx = (z_old - z[x_idx_old - idx_dir, y_idx_old]) / (x[x_idx_old] - x[x_idx_old - idx_dir])
        dx = x[x_idx_new] - x[x_idx_old]
        z_new = max(float(z_old + dz_dx * dx), z_min)
    else:
        # Extrapolate Corner
        dz_x = z[x_idx_new, y_idx_old] - z_old
        dz_y = z[x_idx_old, y_idx_new] - z_old
        z_new = max(float(z_old + dz_x + dz_y), z_min)

    return z_new


data_thrust = data_thrust_original.copy()
x_y_types = ((1, 8, 'ur'),
             (1, 9, 'ur'),
             (3, 10, 'ur'),
             (2, 10, 'ur'),
             (1, 10, 'ur'),
             (0, 2, 'ur'),
             (0, 3, 'ur'),
             (0, 4, 'ur'),
             (0, 5, 'ur'),
             (0, 6, 'ur'),
             (0, 7, 'ur'),
             (0, 8, 'ur'),
             (0, 9, 'ur'),
             (0, 10, 'ur'),
             (7, 1, 'll'),
             (8, 3, 'll'),
             (8, 2, 'll'),
             (8, 1, 'll'),
             (9, 5, 'll'),
             (9, 4, 'll'),
             (9, 3, 'll'),
             (9, 2, 'll'),
             (9, 1, 'll'),
             (0, 0, 'l'),
             (1, 0, 'll'),
             (2, 0, 'll'),
             (3, 0, 'll'),
             (4, 0, 'll'),
             (5, 0, 'll'),
             (6, 0, 'll'),
             (7, 0, 'll'),
             (8, 0, 'll'),
             (9, 0, 'll'),
             )

for x_y_type in x_y_types:
    x_idx_i, y_idx_i, type_i = x_y_type
    if type_i == 'ur':  # upper right
        data_thrust[x_idx_i, y_idx_i] = extrapolate(M_grid_thrust, h_grid_thrust, data_thrust,
                                                    x_idx_i, y_idx_i, x_idx_i + 1, y_idx_i - 1)
    elif type_i == 'll':  # lower left
        data_thrust[x_idx_i, y_idx_i] = extrapolate(M_grid_thrust, h_grid_thrust, data_thrust,
                                                    x_idx_i, y_idx_i, x_idx_i - 1, y_idx_i + 1)
    elif type_i == 'l':  # left
        data_thrust[x_idx_i, y_idx_i] = extrapolate(M_grid_thrust, h_grid_thrust, data_thrust,
                                                    x_idx_i, y_idx_i, x_idx_i, y_idx_i + 1)

# data_thrust = np.hstack((data_thrust[:, 0].reshape(-1, 1), data_thrust))
# data_thrust[:, 0] = data_thrust[:, 1]
data_flat_thrust = data_thrust.ravel(order='F')
thrust_table_bspline = ca.interpolant('thrust_table', 'bspline', (M_grid_thrust, h_grid_thrust), data_flat_thrust)
thrust_table_linear = ca.interpolant('thrust_table', 'linear', (M_grid_thrust, h_grid_thrust), data_flat_thrust)

thrust_input = ca.vcat((M, h))
thrust_bspline = thrust_table_bspline(thrust_input)
thrust_linear = thrust_table_linear(ca.vcat((M, h)))

diff_thrust_fun_bspline = ca.Function('diff_thrust', (M, h),
                                      (ca.jacobian(thrust_bspline, M), ca.jacobian(thrust_bspline, h)),
                                      ('v', 'h'), ('dT_dv', 'dT_dh'))
diff_thrust_fun_linear = ca.Function('diff_thrust', (M, h),
                                     (ca.jacobian(thrust_linear, M), ca.jacobian(thrust_linear, h)),
                                     ('v', 'h'), ('dT_dv', 'dT_dh'))

M_grid_aero = np.array((0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8))
data_CLalpha = np.array((3.44, 3.44, 3.44, 3.44, 3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44))
data_CD0 = np.array((0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035))
data_eta = np.array((0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93))

CLalpha_table_bspline = ca.interpolant('CLalpha_table', 'bspline', (M_grid_aero,), data_CLalpha)
CD0_table_bspline = ca.interpolant('CD0_table', 'bspline', (M_grid_aero,), data_CD0)
eta_table_bspline = ca.interpolant('eta_table', 'bspline', (M_grid_aero,), data_eta)

CLalpha_table_linear = ca.interpolant('CLalpha_table', 'linear', (M_grid_aero,), data_CLalpha)
CD0_table_linear = ca.interpolant('CD0_table', 'linear', (M_grid_aero,), data_CD0)
eta_table_linear = ca.interpolant('eta_table', 'linear', (M_grid_aero,), data_eta)

CLalpha_bspline = CLalpha_table_bspline(M)
CD0_bspline = CD0_table_bspline(M)
eta_bspline = eta_table_bspline(M)

CLalpha_linear = CLalpha_table_linear(M)
CD0_linear = CD0_table_linear(M)
eta_linear = eta_table_linear(M)

diff_CLalpha_fun_bspline = ca.Function('dCLalpha_dv',
                                       (M,), (ca.jacobian(CLalpha_bspline, M),), ('v',), ('dCLalpha_dv',))
diff_CD0_fun_bspline = ca.Function('dCD0_dv', (M,), (ca.jacobian(CD0_bspline, M),), ('v',), ('dCD0_dv',))
diff_eta_fun_bspline = ca.Function('deta_dv', (M,), (ca.jacobian(eta_bspline, M),), ('v',), ('deta_dv',))

diff_CLalpha_fun_linear = ca.Function('dCLalpha_dv', (M,), (ca.jacobian(CLalpha_linear, M),), ('v',), ('dCLalpha_dv',))
diff_CD0_fun_linear = ca.Function('dCD0_dv', (M,), (ca.jacobian(CD0_linear, M),), ('v',), ('dCD0_dv',))
diff_eta_fun_linear = ca.Function('deta_dv', (M,), (ca.jacobian(eta_linear, M),), ('v',), ('deta_dv',))

# Expand Table for flatter subsonic spline
# Added Points: 0.2, 0.6, 0.7, 0.79 all flat
# Optimize intermediate values to minimize curvature
M_grid_aero_expanded = np.array((0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.825, 0.875, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8))
# M_grid_aero_expanded = np.array((0, 0.2, 0.4, 0.6, 0.7, 0.79, 0.8, 0.825, 0.875, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8))

atm = Atmosphere1976(use_metric=False)
# vals_per_layer = 10
# h_buffer = 1_000  # ft
# h_grid_atm = np.concatenate((np.linspace(atm.h_layers[0], atm.h_layers[1] - h_buffer, vals_per_layer),
#                              np.linspace(atm.h_layers[1] + h_buffer, atm.h_layers[2] - h_buffer, vals_per_layer),
#                              np.linspace(atm.h_layers[2] + h_buffer, atm.h_layers[3], vals_per_layer)))
# h_grid_atm = h_grid_thrust
h_grid_atm = np.array((-2, 0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 71)) * 1e3
# h_grid_atm = np.array((-2, 0, 5, 10, 15, 20, 25, 30, 40, 45, 50, 65, 70)) * 1e3
data_temp = np.asarray([atm.temperature(alt) for alt in h_grid_atm])
data_sond = np.asarray([atm.speed_of_sound(alt) for alt in h_grid_atm])
data_dens = np.asarray([atm.density(alt) for alt in h_grid_atm])

temp_table_bspline = ca.interpolant('T', 'bspline', (h_grid_atm,), data_temp)
sond_table_bspline = ca.interpolant('a', 'bspline', (h_grid_atm,), data_sond)
dens_table_bspline = ca.interpolant('T', 'bspline', (h_grid_atm,), data_dens)

temp_bspline = temp_table_bspline(h)
sond_bspline = sond_table_bspline(h)
dens_bspline = dens_table_bspline(h)

diff_temp_fun_bspline = ca.Function('dT_dh', (h,), (ca.jacobian(temp_bspline, h),), ('h',), ('dT_dh',))
diff_sond_fun_bspline = ca.Function('da_dh', (h,), (ca.jacobian(sond_bspline, h),), ('h',), ('dT_dh',))
diff_dens_fun_bspline = ca.Function('drho_dh', (h,), (ca.jacobian(dens_bspline, h),), ('h',), ('drho_dh',))

CLalpha_table_pchip = PchipInterpolator(M_grid_aero, data_CLalpha)
CD0_table_pchip = PchipInterpolator(M_grid_aero, data_CD0)
eta_table_pchip = PchipInterpolator(M_grid_aero, data_eta)

data_CLalpha_expanded = CLalpha_table_pchip(M_grid_aero_expanded)
data_CD0_expanded = CD0_table_pchip(M_grid_aero_expanded)
data_eta_expanded = eta_table_pchip(M_grid_aero_expanded)

CLalpha_table_bspline_expanded = ca.interpolant('CLalpha_table', 'bspline',
                                                (M_grid_aero_expanded,), data_CLalpha_expanded)
CD0_table_bspline_expanded = ca.interpolant('CLalpha_table', 'bspline', (M_grid_aero_expanded,), data_CD0_expanded)
eta_table_bspline_expanded = ca.interpolant('CLalpha_table', 'bspline', (M_grid_aero_expanded,), data_eta_expanded)

CLalpha_bspline_expanded = CLalpha_table_bspline_expanded(M)
CD0_bspline_expanded = CD0_table_bspline_expanded(M)
eta_bspline_expanded = eta_table_bspline_expanded(M)

diff_CLalpha_fun_bspline_expanded = ca.Function('dCLalpha_dM', (M,), (ca.jacobian(CLalpha_bspline_expanded, M),),
                                                ('M',), ('dCLalpha_dM',))
diff_CD0_fun_bspline_expanded = ca.Function('dCD0_dM', (M,), (ca.jacobian(CD0_bspline_expanded, M),),
                                            ('M',), ('dCD0_dM',))
diff_eta_fun_bspline_expanded = ca.Function('deta_dM', (M,), (ca.jacobian(eta_bspline_expanded, M),),
                                            ('M',), ('deta_dM',))

# noinspection PyTypeChecker
np.savetxt('lut_aero_data.csv', np.hstack((M_grid_aero_expanded.reshape((-1, 1)),
                                           data_CLalpha_expanded.reshape((-1, 1)),
                                           data_CD0_expanded.reshape((-1, 1)),
                                           data_eta_expanded.reshape((-1, 1)),)), delimiter=',')

# noinspection PyTypeChecker
np.savetxt('lut_atm_data.csv', np.hstack((h_grid_atm.reshape((-1, 1)),
                                           data_dens.reshape((-1, 1)),
                                           data_sond.reshape((-1, 1)),)), delimiter=',')

# noinspection PyTypeChecker
np.savetxt('lut_thrust_data.csv', np.vstack((
    np.hstack((np.zeros(shape=(1, 1)), h_grid_thrust.reshape((1, -1)),)),
    np.hstack((M_grid_thrust.reshape((-1, 1)), data_thrust,)),
)))

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    M_LAB = 'Mach'
    N_VALS = 1_000

    MED_FIG_SIZE = (6.5, 5)
    SML_FIG_SIZE = (6.5, 3)

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gradient = mpl.colormaps['viridis'].colors
    grad_idcs = np.int32(np.ceil(np.linspace(0, 255, len(h_grid_thrust))))

    def cols_gradient(n):
        return gradient[grad_idcs[n]]

    M = np.linspace(0, 1.8, N_VALS)  # Mach number
    M_2D = M.reshape(1, -1)
    h = np.linspace(h_grid_thrust[0], h_grid_thrust[-1], N_VALS)  # Altitude
    h_atm = np.linspace(h_grid_atm[0] + 1, h_grid_atm[-1], N_VALS)

    expanded_idcs = []
    for idx, m_val in enumerate(M_grid_aero_expanded):
        if not any(m_val == M_grid_aero):
            expanded_idcs.append(idx)

    expanded_idcs = (tuple(expanded_idcs),)

    CLalpha_bspline_vals = CLalpha_table_bspline(M)
    CD0_bspline_vals = CD0_table_bspline(M)
    eta_bspline_vals = eta_table_bspline(M)

    CLalpha_bspline_expanded_vals = CLalpha_table_bspline_expanded(M)
    CD0_bspline_expanded_vals = CD0_table_bspline_expanded(M)
    eta_bspline_expanded_vals = eta_table_bspline_expanded(M)

    CLalpha_linear_vals = CLalpha_table_linear(M)
    CD0_linear_vals = CD0_table_linear(M)
    eta_linear_vals = eta_table_linear(M)

    temp_vals = temp_table_bspline(h_atm)
    sond_vals = sond_table_bspline(h_atm)
    dens_vals = dens_table_bspline(h_atm)

    h_sym = ca.SX.sym('h')
    temp_sx, _, dens_sx = atm.get_ca_atm_expr(h_sym)
    sond_sx = (temp_sx * atm.gas_constant * atm.specific_heat_ratio) ** 0.5
    dtemp_sx = ca.jacobian(temp_sx, h_sym)
    dsond_sx = ca.jacobian(sond_sx, h_sym)
    ddens_sx = ca.jacobian(dens_sx, h_sym)

    temp_cond_func = ca.Function('T', (h_sym,), (temp_sx,), ('h',), ('T',))
    sond_cond_func = ca.Function('a', (h_sym,), (sond_sx,), ('h',), ('a',))
    dens_cond_func = ca.Function('rho', (h_sym,), (dens_sx,), ('h',), ('rho',))
    dtemp_cond_func = ca.Function('dT', (h_sym,), (dtemp_sx,), ('h',), ('dT',))
    dsond_cond_func = ca.Function('da', (h_sym,), (dsond_sx,), ('h',), ('da',))
    ddens_cond_func = ca.Function('drho', (h_sym,), (ddens_sx,), ('h',), ('drho',))

    temp_cond_vals = np.asarray(temp_cond_func(h_atm))
    sond_cond_vals = np.asarray(sond_cond_func(h_atm))
    dens_cond_vals = np.asarray(dens_cond_func(h_atm))
    dtemp_cond_vals = np.asarray(dtemp_cond_func(h_atm))
    dsond_cond_vals = np.asarray(dsond_cond_func(h_atm))
    ddens_cond_vals = np.asarray(ddens_cond_func(h_atm))

    dCLalpha_bspline_dM = diff_CLalpha_fun_bspline(M)
    dCD0_bspline_dM = diff_CD0_fun_bspline(M)
    deta_bspline_dM = diff_eta_fun_bspline(M)

    dCLalpha_bspline_dM_expanded = diff_CLalpha_fun_bspline_expanded(M)
    dCD0_bspline_dM_expanded = diff_CD0_fun_bspline_expanded(M)
    deta_bspline_dM_expanded = diff_eta_fun_bspline_expanded(M)

    dCLalpha_linear_dM = diff_CLalpha_fun_linear(M)
    dCD0_linear_dM = diff_CD0_fun_linear(M)
    deta_linear_dM = diff_eta_fun_linear(M)

    dTemp_dh = diff_temp_fun_bspline(h_atm)
    dSond_dh = diff_sond_fun_bspline(h_atm)
    dDens_dh = diff_dens_fun_bspline(h_atm)

    # FIGURE 1 (CLalpha)
    fig1 = plt.figure(figsize=SML_FIG_SIZE)

    ax11 = fig1.add_subplot(211)
    ax11.plot(M, CLalpha_linear_vals, color=cols[1], label='Linear')
    ax11.plot(M, CLalpha_bspline_vals, color=cols[0], label='Spline')
    ax11.plot(M, CLalpha_bspline_expanded_vals, '--', color=cols[0])
    ax11.plot(M_grid_aero, data_CLalpha, 'kx', label='Table')
    ax11.plot(M_grid_aero_expanded[expanded_idcs], data_CLalpha_expanded[expanded_idcs], 'ko')
    ax11.grid()
    ax11.set_ylabel(r'$C_{L,\alpha}$')

    ax12 = fig1.add_subplot(212)
    ax12.plot(M, dCLalpha_linear_dM, color=cols[1])
    ax12.plot(M, dCLalpha_bspline_dM, color=cols[0])
    ax12.plot(M, dCLalpha_bspline_dM_expanded, '--', color=cols[0])
    ax12.grid()
    ax12.set_ylabel(r'$\dfrac{dC_{L,\alpha}}{dM}$')
    ax12.set_xlabel(M_LAB)

    fig1.tight_layout()

    # FIGURE 2 (CD0)
    fig2 = plt.figure(figsize=SML_FIG_SIZE)

    ax21 = fig2.add_subplot(211)
    ax21.plot(M, CD0_linear_vals, color=cols[1], label='Linear')
    ax21.plot(M, CD0_bspline_vals, color=cols[0], label='Spline')
    ax21.plot(M, CD0_bspline_expanded_vals, '--', color=cols[0])
    ax21.plot(M_grid_aero, data_CD0, 'kx', label='Table')
    ax21.plot(M_grid_aero_expanded[expanded_idcs], data_CD0_expanded[expanded_idcs], 'ko')
    ax21.grid()
    ax21.set_ylabel(r'$C_{D,0}$')

    ax22 = fig2.add_subplot(212)
    ax22.plot(M, dCD0_linear_dM, color=cols[1])
    ax22.plot(M, dCD0_bspline_dM, color=cols[0])
    ax22.plot(M, dCD0_bspline_dM_expanded, '--', color=cols[0])
    ax22.grid()
    ax22.set_ylabel(r'$\dfrac{dC_{D,0}}{dM}$')
    ax22.set_xlabel(M_LAB)

    fig2.tight_layout()

    # FIGURE 3 (Eta)
    fig3 = plt.figure(figsize=SML_FIG_SIZE)

    ax31 = fig3.add_subplot(211)
    ax31.plot(M, eta_linear_vals, color=cols[1], label='Linear')
    ax31.plot(M, eta_bspline_vals, color=cols[0], label='Spline')
    ax31.plot(M, eta_bspline_expanded_vals, '--', color=cols[0])
    ax31.plot(M_grid_aero, data_eta, 'kx', label='Table')
    ax31.plot(M_grid_aero_expanded[expanded_idcs], data_eta_expanded[expanded_idcs], 'ko')
    ax31.grid()
    ax31.set_ylabel(r'$\eta$')

    ax32 = fig3.add_subplot(212)
    ax32.plot(M, deta_linear_dM, color=cols[1])
    ax32.plot(M, deta_bspline_dM, color=cols[0])
    ax32.plot(M, deta_bspline_dM_expanded, '--', color=cols[0])
    ax32.grid()
    ax32.set_ylabel(r'$\dfrac{d\eta}{dM}$')
    ax32.set_xlabel(M_LAB)

    fig3.tight_layout()

    # FIGURE 4 (Thrust)
    fig4 = plt.figure(figsize=MED_FIG_SIZE)
    ax41 = fig4.add_subplot(311)
    ax42 = fig4.add_subplot(312)
    ax43 = fig4.add_subplot(313)

    for idx, alt in enumerate(h_grid_thrust):
        thrust_bspline_vals = []
        thrust_linear_vals = []
        dT_dM_vals_bspline = []
        dT_dh_vals_bspline = []
        dT_dM_vals_linear = []
        dT_dh_vals_linear = []

        for M_val in M:
            thrust_bspline_vals.append(thrust_table_bspline(np.vstack((M_val, alt))))
            thrust_linear_vals.append(thrust_table_linear(np.vstack((M_val, alt))))

            dT_dM_bspline, dT_dh_bspline = diff_thrust_fun_bspline(M_val, alt)
            dT_dM_linear, dT_dh_linear = diff_thrust_fun_linear(M_val, alt)

            dT_dM_vals_bspline.append(dT_dM_bspline)
            dT_dh_vals_bspline.append(dT_dh_bspline)
            dT_dM_vals_linear.append(dT_dM_linear)
            dT_dh_vals_linear.append(dT_dh_linear)

        thrust_bspline_vals = np.asarray(thrust_bspline_vals).flatten()
        thrust_linear_vals = np.asarray(thrust_linear_vals).flatten()
        dT_dM_vals_bspline = np.asarray(dT_dM_vals_bspline).flatten()
        dT_dh_vals_bspline = np.asarray(dT_dh_vals_bspline).flatten()
        dT_dM_vals_linear = np.asarray(dT_dM_vals_linear).flatten()
        dT_dh_vals_linear = np.asarray(dT_dh_vals_linear).flatten()

        extrapolated_idcs = np.where(np.isnan(data_thrust_original[:, idx]))

        ax41.plot(M, thrust_linear_vals / 10_000, color=cols_gradient(idx))
        ax41.plot(M, thrust_bspline_vals / 10_000, '--', color=cols_gradient(idx))
        ax41.plot(M_grid_thrust, data_thrust_original[:, idx] / 10_000, 'x', color=cols_gradient(idx))
        ax41.plot(M_grid_thrust[extrapolated_idcs],
                  data_thrust[extrapolated_idcs, idx].flatten() / 10_000,
                  'o', color=cols_gradient(idx))

        ax42.plot(M, dT_dM_vals_linear / 10_000, color=cols_gradient(idx), label='Linear')
        ax42.plot(M, dT_dM_vals_bspline / 10_000, '--', color=cols_gradient(idx), label='Spline')

        if alt == h_grid_thrust[0]:
            ax43.plot(M, dT_dh_vals_linear, color=cols_gradient(idx), label='h = 0 ft')

        elif alt == h_grid_thrust[-1]:
            ax43.plot(M, dT_dh_vals_linear, color=cols_gradient(idx), label='h = 70,000 ft')
        else:
            ax43.plot(M, dT_dh_vals_linear, color=cols_gradient(idx))
        ax43.plot(M, dT_dh_vals_bspline, '--', color=cols_gradient(idx))

    ax41.grid()
    ax42.grid()
    ax43.grid()
    # ax43.legend(loc='best')

    ax41.set_ylabel(r'($T_{hrust}$) [10,000 lb]')
    ax42.set_ylabel(r'$\dfrac{\partial T_{hrust}}{\partial M}$ [10,000 lb]')
    ax43.set_ylabel(r'$\dfrac{\partial T_{hrust}}{\partial h}$ [lb/ft]')
    ax43.set_xlabel(M_LAB)

    # FIGURE 5 (ATMOSPHERE)
    fig5 = plt.figure(figsize=(6.5, 5))

    ax51 = fig5.add_subplot(231)
    ax51.plot(h_atm / 1_000, temp_cond_vals, label='Cond.')
    ax51.plot(h_atm / 1_000, temp_vals, label='Spline', zorder=0)
    ax51.plot(h_grid_atm / 1_000, data_temp, 'kx', label='1976 Atm Data')
    ax51.grid()
    ax51.legend()
    ax51.set_ylabel(r'Temp ($T$) [deg R]')

    ax51 = fig5.add_subplot(232)
    ax51.plot(h_atm / 1_000, sond_cond_vals, label='Cond.')
    ax51.plot(h_atm / 1_000, sond_vals, label='Spline', zorder=0)
    ax51.plot(h_grid_atm / 1_000, data_sond, 'kx', label='1976 Atm Data')
    ax51.grid()
    ax51.legend()
    ax51.set_ylabel(r'Sp. of Sound ($a$) [ft/s]')

    ax52 = fig5.add_subplot(233)
    ax52.plot(h_atm / 1_000, dens_cond_vals, label='Cond.')
    ax52.plot(h_atm / 1_000, dens_vals, label='Spline', zorder=0)
    ax52.plot(h_grid_atm / 1_000, data_dens, 'kx')
    ax52.grid()
    ax52.set_yscale('log')
    ax52.set_ylabel(r'Dens ($\rho$) [slug/ft$^3$]')

    ax53 = fig5.add_subplot(234)
    ax53.plot(h_atm / 1_000, dtemp_cond_vals, label='Cond.')
    ax53.plot(h_atm / 1_000, dTemp_dh, label='Spline', zorder=0)
    ax53.grid()
    ax53.set_ylabel(r'$\dfrac{dT}{dh}$ [deg R/ft]')
    ax53.set_xlabel('h [1,000 ft]')

    ax53 = fig5.add_subplot(235)
    ax53.plot(h_atm / 1_000, dsond_cond_vals, label='Cond.')
    ax53.plot(h_atm / 1_000, dSond_dh, label='Spline', zorder=0)
    ax53.grid()
    ax53.set_ylabel(r'$\dfrac{da}{dh}$ [1/s]')
    ax53.set_xlabel('h [1,000 ft]')

    ax54 = fig5.add_subplot(236)
    ax54.plot(h_atm / 1_000, ddens_cond_vals, label='Cond.')
    ax54.plot(h_atm / 1_000, dDens_dh, label='Spline', zorder=0)
    ax54.grid()
    ax54.set_ylabel(r'$\dfrac{d\rho}{dh}$ [slug/ft$^4$]')
    ax54.set_xlabel('h [1,000 ft]')

    fig5.tight_layout()

    # SAVE FIGURES
    fig1.savefig('lut_CLalpha.eps',
                 format='eps',
                 bbox_inches='tight')

    fig2.savefig('lut_CD0.eps',
                 format='eps',
                 bbox_inches='tight')

    fig3.savefig('lut_eta.eps',
                 format='eps',
                 bbox_inches='tight')

    fig4.savefig('lut_thrust.eps',
                 format='eps',
                 bbox_inches='tight')

    fig5.savefig('lut_atmosphere.eps',
                 format='eps',
                 bbox_inches='tight')

    plt.show()
