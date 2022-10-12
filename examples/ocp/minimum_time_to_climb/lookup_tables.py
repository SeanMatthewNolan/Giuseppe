import casadi as ca
import numpy as np

a = 1125.33  # speed of sound [ft/s]
v = ca.MX.sym('v', 1)
h = ca.MX.sym('h', 1)
M = v / a  # assume a = 343 m/s = 1125.33 ft/s
interp_method = 'bspline'  # either 'bspline' or 'linear'

M_grid_thrust = np.array((0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8))
h_grid_thrust = np.array((0, 5, 10, 15, 20, 25, 30, 40, 50, 70)) * 1e3

data_thrust_original = np.array(((24.2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
                                 (28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, np.nan, np.nan, np.nan),
                                 (28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, np.nan),
                                 (30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, np.nan),
                                 (34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1),
                                 (37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4),
                                 (36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7),
                                 (np.nan, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0, 2.2),
                                 (np.nan, np.nan, np.nan, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9),
                                 (np.nan, np.nan, np.nan, np.nan, np.nan, 34.6, 31.1, 21.7, 13.3, 3.1))) * 1e3

data_thrust = np.array(((24.2, 24.4, 22.75, 20.4, 18.8, 15.8, 13.3, 11.2, 9.0, 7.1),
                        (28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, 9.0, 6.7, 5.2),
                        (28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, 3.7),
                        (30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, 3.0),
                        (34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1),
                        (37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4),
                        (36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7),
                        (36.4, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0, 2.2),
                        (37.0, 37.6, 38.6, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9),
                        (37.4, 37.7, 37.8, 37.0, 35.2, 34.6, 31.1, 21.7, 13.3, 3.1))) * 1e3

data_flat_thrust = data_thrust.ravel(order='F')
thrust_table_bspline = ca.interpolant('thrust_table', 'bspline', (M_grid_thrust, h_grid_thrust), data_flat_thrust)
thrust_table_linear = ca.interpolant('thrust_table', 'linear', (M_grid_thrust, h_grid_thrust), data_flat_thrust)

thrust_input = ca.vcat((M, h))
thrust_bspline = thrust_table_bspline(thrust_input)
thrust_linear = thrust_table_linear(ca.vcat((M, h)))

diff_thrust_fun_bspline = ca.Function('diff_thrust', (v, h),
                                      (ca.jacobian(thrust_bspline, v), ca.jacobian(thrust_bspline, h)),
                                      ('v', 'h'), ('dT_dv', 'dT_dh'))
diff_thrust_fun_linear = ca.Function('diff_thrust', (v, h),
                                     (ca.jacobian(thrust_linear, v), ca.jacobian(thrust_linear, h)),
                                     ('v', 'h'), ('dT_dv', 'dT_dh'))

M_grid_aero = np.array((0, 0.4, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8))
data_CLalpha = np.array((3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44))
data_CD0 = np.array((0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035))
data_eta = np.array((0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93))

CLalpha_table_bspline = ca.interpolant('CLalpha_table', 'bspline', (M_grid_aero,), data_CLalpha)
CD0_table_bspline = ca.interpolant('CLalpha_table', 'bspline', (M_grid_aero,), data_CD0)
eta_table_bspline = ca.interpolant('CLalpha_table', 'bspline', (M_grid_aero,), data_eta)

CLalpha_table_linear = ca.interpolant('CLalpha_table', 'linear', (M_grid_aero,), data_CLalpha)
CD0_table_linear = ca.interpolant('CLalpha_table', 'linear', (M_grid_aero,), data_CD0)
eta_table_linear = ca.interpolant('CLalpha_table', 'linear', (M_grid_aero,), data_eta)

CLalpha_bspline = CLalpha_table_bspline(M)
CD0_bspline = CD0_table_bspline(M)
eta_bspline = eta_table_bspline(M)

CLalpha_linear = CLalpha_table_linear(M)
CD0_linear = CD0_table_linear(M)
eta_linear = eta_table_linear(M)

diff_CLalpha_fun_bspline = ca.Function('dCLalpha_dv', (v,), (CLalpha_bspline,), ('v',), ('dCLalpha_dv',))
diff_CD0_fun_bspline = ca.Function('dCD0_dv', (v,), (CD0_bspline,), ('v',), ('dCD0_dv',))
diff_eta_fun_bspline = ca.Function('deta_dv', (v,), (ca.jacobian(eta_bspline, v),), ('v',), ('deta_dv',))

diff_CLalpha_fun_linear = ca.Function('dCLalpha_dv', (v,), (CLalpha_linear,), ('v',), ('dCLalpha_dv',))
diff_CD0_fun_linear = ca.Function('dCD0_dv', (v,), (CD0_linear,), ('v',), ('dCD0_dv',))
diff_eta_fun_linear = ca.Function('deta_dv', (v,), (ca.jacobian(eta_linear, v),), ('v',), ('deta_dv',))

# Expand Table for flatter subsonic spline
M_grid_aero_expanded = np.array((0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8))
data_CLalpha_expanded = np.array((3.44, 3.44, 3.44, 3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44))
data_CD0_expanded = np.array((0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035))
data_eta_expanded = np.array((0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93))

CLalpha_table_bspline_expanded = ca.interpolant('CLalpha_table', 'bspline',
                                                (M_grid_aero_expanded,), data_CLalpha_expanded)
CD0_table_bspline_expanded = ca.interpolant('CLalpha_table', 'bspline', (M_grid_aero_expanded,), data_CD0_expanded)
eta_table_bspline_expanded = ca.interpolant('CLalpha_table', 'bspline', (M_grid_aero_expanded,), data_eta_expanded)

eta_bspline_expanded = eta_table_bspline_expanded(M)

diff_eta_fun_bspline_expanded = ca.Function('deta_dv', (v,), (ca.jacobian(eta_bspline_expanded, v),),
                                            ('v',), ('deta_dv',))

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    M_LAB = 'Mach'
    N_VALS = 100

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gradient = mpl.colormaps['viridis'].colors
    grad_idcs = np.int32(np.ceil(np.linspace(0, 255, len(h_grid_thrust))))

    def cols_gradient(n):
        return gradient[grad_idcs[n]]

    M = np.linspace(0, 1.8, N_VALS)  # Mach number
    M_2D = M.reshape(1, -1)
    v = M * a  # Velocity
    h = np.linspace(0, 70_000, N_VALS)  # Altitude

    expanded_idcs = []
    for idx, m_val in enumerate(M_grid_aero_expanded):
        if not any(m_val == M_grid_aero):
            expanded_idcs.append(idx)

    expanded_idcs = (tuple(expanded_idcs),)

    eta_bspline_vals = eta_table_bspline(M)
    CLalpha_bspline_vals = CLalpha_table_bspline(M)
    CD0_bspline_vals = CD0_table_bspline(M)
    eta_bspline_expanded_vals = eta_table_bspline_expanded(M)

    eta_linear_vals = eta_table_linear(M)
    CLalpha_linear_vals = CLalpha_table_linear(M)
    CD0_linear_vals = CD0_table_linear(M)
    eta_linear_expanded_vals = eta_table_linear(M)

    dv_dM = a

    diff_eta_bspline_vals = diff_eta_fun_bspline(v)
    diff_CLalpha_bspline_vals = diff_CLalpha_fun_bspline(v)
    diff_CD0_bspline_vals = diff_CD0_fun_bspline(v)
    diff_eta_bspline_expanded_vals = diff_eta_fun_bspline_expanded(v)

    diff_eta_linear_vals = diff_eta_fun_linear(v)
    diff_CLalpha_linear_vals = diff_CLalpha_fun_linear(v)
    diff_CD0_linear_vals = diff_CD0_fun_linear(v)

    deta_bspline_dM = diff_eta_bspline_vals * dv_dM
    dCLalpha_bspline_dM = diff_CLalpha_bspline_vals * dv_dM
    dCD0_bspline_dM = diff_CD0_bspline_vals * dv_dM
    deta_bspline_dM_expanded = diff_eta_bspline_expanded_vals * dv_dM

    deta_linear_dM = diff_eta_linear_vals * dv_dM
    dCLalpha_linear_dM = diff_CLalpha_linear_vals * dv_dM
    dCD0_linear_dM = diff_CD0_linear_vals * dv_dM

    # FIGURE 1 (Eta)
    fig1 = plt.figure(figsize=(6.5, 5))

    ax11 = fig1.add_subplot(211)
    ax11.plot(M, eta_linear_vals, color=cols[1], label='Linear')
    ax11.plot(M, eta_bspline_vals, color=cols[0], label='Spline')
    ax11.plot(M, eta_bspline_expanded_vals, '--', color=cols[0])
    ax11.plot(M_grid_aero, data_eta, 'kx', label='Table')
    ax11.plot(M_grid_aero_expanded[expanded_idcs], data_eta_expanded[expanded_idcs], 'ko')
    ax11.grid()
    ax11.set_ylabel(r'$\eta$')

    ax12 = fig1.add_subplot(212)
    ax12.plot(M, deta_linear_dM, color=cols[1])
    ax12.plot(M, deta_bspline_dM, color=cols[0])
    ax12.plot(M, deta_bspline_dM_expanded, '--', color=cols[0])
    ax12.grid()
    ax12.set_ylabel(r'$\dfrac{d\eta}{dM}$')
    ax12.set_xlabel(M_LAB)

    fig1.tight_layout()

    # FIGURE 2 (CLalpha)
    fig2 = plt.figure(figsize=(6.5, 5))

    ax21 = fig2.add_subplot(211)
    ax21.plot(M, CLalpha_linear_vals, color=cols[1], label='Linear')
    ax21.plot(M, CLalpha_bspline_vals, color=cols[0], label='Spline')

    ax21.plot(M_grid_aero, data_CLalpha, 'kx', label='Table')
    ax21.grid()
    ax21.set_ylabel(r'$C_{L,\alpha}$')

    ax22 = fig2.add_subplot(212)
    ax22.plot(M, dCLalpha_linear_dM, color=cols[1])
    ax22.plot(M, dCLalpha_bspline_dM, color=cols[0])
    ax22.grid()
    ax22.set_ylabel(r'$\dfrac{dC_{L,\alpha}}{dM}$')
    ax22.set_xlabel(M_LAB)

    fig2.tight_layout()

    # FIGURE 3 (CD0)
    fig3 = plt.figure(figsize=(6.5, 5))

    ax31 = fig3.add_subplot(211)
    ax31.plot(M, CD0_linear_vals, color=cols[1], label='Linear')
    ax31.plot(M, CD0_bspline_vals, color=cols[0], label='Spline')
    ax31.plot(M_grid_aero, data_CD0, 'kx', label='Table')
    ax31.grid()
    ax31.set_ylabel(r'$C_{D,0}$')

    ax32 = fig3.add_subplot(212)
    ax32.plot(M, dCD0_linear_dM, color=cols[1])
    ax32.plot(M, dCD0_bspline_dM, color=cols[0])
    ax32.grid()
    ax32.set_ylabel(r'$\dfrac{dC_{D,0}}{dM}$')
    ax32.set_xlabel(M_LAB)

    fig3.tight_layout()

    # FIGURE 4 (Thrust)
    fig4 = plt.figure(figsize=(6.5, 5))
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

            dT_dv_bspline, dT_dh_bspline = diff_thrust_fun_bspline(M_val * a, alt)
            dT_dv_linear, dT_dh_linear = diff_thrust_fun_linear(M_val * a, alt)

            dT_dM_vals_bspline.append(dT_dv_bspline * dv_dM)
            dT_dh_vals_bspline.append(dT_dh_bspline)
            dT_dM_vals_linear.append(dT_dv_linear * dv_dM)
            dT_dh_vals_linear.append(dT_dh_linear)

        thrust_bspline_vals = np.asarray(thrust_bspline_vals).flatten()
        thrust_linear_vals = np.asarray(thrust_linear_vals).flatten()
        dT_dM_vals_bspline = np.asarray(dT_dM_vals_bspline).flatten()
        dT_dh_vals_bspline = np.asarray(dT_dh_vals_bspline).flatten()
        dT_dM_vals_linear = np.asarray(dT_dM_vals_linear).flatten()
        dT_dh_vals_linear = np.asarray(dT_dh_vals_linear).flatten()

        extrapolated_idcs = np.where(np.isnan(data_thrust_original[:, idx]))

        ax41.plot(M, thrust_linear_vals / 10_000, color=cols_gradient(idx), label='Linear')
        ax41.plot(M, thrust_bspline_vals / 10_000, '--', color=cols_gradient(idx), label='Spline')
        ax41.plot(M_grid_thrust, data_thrust_original[:, idx] / 10_000, 'x', color=cols_gradient(idx), label='Table')
        ax41.plot(M_grid_thrust[extrapolated_idcs],
                  data_thrust[extrapolated_idcs, idx].flatten() / 10_000,
                  'o', color=cols_gradient(idx))

        ax42.plot(M, dT_dM_vals_linear / 10_000, color=cols_gradient(idx), label='Linear')
        ax42.plot(M, dT_dM_vals_bspline / 10_000, '--', color=cols_gradient(idx), label='Spline')

        ax43.plot(M, dT_dh_vals_linear, color=cols_gradient(idx), label='Linear')
        ax43.plot(M, dT_dh_vals_bspline, '--', color=cols_gradient(idx), label='Spline')

    ax41.grid()
    ax42.grid()
    ax43.grid()

    ax41.set_ylabel(r'Thrust ($T$) [10,000 lb]')
    ax42.set_ylabel(r'$\dfrac{dT}{dM}$ [10,000 lb]')
    ax43.set_ylabel(r'$\dfrac{dT}{dh}$ [lb/ft]')
    ax43.set_xlabel(M_LAB)

    plt.show()
