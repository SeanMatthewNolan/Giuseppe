import os

import numpy as np

from giuseppe.guess import initialize_guess, propagate_guess, propagate_guess_ocp, propagate_guess_dual
from giuseppe.guess.sequential_linear_projection import match_constants_to_boundary_conditions,\
    match_states_to_boundary_conditions, match_adjoints
from giuseppe.problems.input import StrInputProb
from giuseppe.problems.symbolic import SymDual, SymOCP, SymAdjoints
from giuseppe.problems.conversions import convert_dual_to_bvp
from giuseppe.problems.regularization import PenaltyConstraintHandler

os.chdir(os.path.dirname(__file__))  # Set directory to current location

ocp = StrInputProb()

ocp.set_independent('t')

ocp.add_expression('r', 're + h')
ocp.add_expression('g', 'mu / r**2')
ocp.add_expression('rho', 'rho_0 * exp(-h / h_ref)')
ocp.add_expression('dyn_pres', '1 / 2 * rho * v ** 2 ')
ocp.add_expression('lift', 'c_l * s_ref * dyn_pres')
ocp.add_expression('drag', 'c_d * s_ref * dyn_pres')
ocp.add_expression('c_l', 'a_0 + a_1 * alpha_hat')
ocp.add_expression('c_d', 'b_0 + b_1 * alpha_hat + b_2 * alpha_hat**2')
ocp.add_expression('alpha_hat', 'alpha * 180 / pi')

ocp.add_state('h', 'v * sin(gamma)')
ocp.add_state('phi', 'v * cos(gamma) * sin(psi) / (r * cos(theta))')
ocp.add_state('theta', 'v * cos(gamma) * cos(psi) / r')
ocp.add_state('v', '-drag / m - g * sin(gamma)')
ocp.add_state('gamma', 'lift * cos(beta) / (m * v) + cos(gamma) * (v / r - g / v)')
ocp.add_state('psi', 'lift * sin(beta)/(m * v * cos(gamma)) + v * cos(gamma) * sin(psi) * sin(theta)/(r * cos(theta))')

ocp.add_control('alpha')
ocp.add_control('beta')

ocp.add_constant('rho_0', 0.002378)
ocp.add_constant('h_ref', 23_800)
ocp.add_constant('re', 20_902_900)
ocp.add_constant('m', 203_000 / 32.174)
ocp.add_constant('mu', 0.14076539e17)

ocp.add_constant('a_0', -0.20704)
ocp.add_constant('a_1', 0.029244)
ocp.add_constant('b_0', 0.07854)
ocp.add_constant('b_1', -0.61592e-2)
ocp.add_constant('b_2', 0.621408e-3)
ocp.add_constant('s_ref', 2690)

ocp.add_constant('xi', 0)

ocp.add_constant('eps_alpha', 1e-5)
ocp.add_constant('alpha_min', -80 / 180 * 3.1419)
ocp.add_constant('alpha_max', 80 / 180 * 3.1419)

ocp.add_constant('eps_beta', 1e-10)
ocp.add_constant('beta_min', -85 / 180 * 3.1419)
ocp.add_constant('beta_max', 85 / 180 * 3.1419)

ocp.add_constant('h_0', 260_000)
ocp.add_constant('phi_0', 0)
ocp.add_constant('theta_0', 0)
ocp.add_constant('v_0', 25_600)
ocp.add_constant('gamma_0', -1 / 180 * np.pi)
ocp.add_constant('psi_0', np.pi / 2)

ocp.add_constant('h_f', 80_000)
ocp.add_constant('v_f', 2_500)
ocp.add_constant('gamma_f', -5 / 180 * np.pi)

ocp.set_cost('0', '0', '-phi * cos(xi) - theta  * sin(xi)')

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'h - h_0')
ocp.add_constraint('initial', 'phi - phi_0')
ocp.add_constraint('initial', 'theta - theta_0')
ocp.add_constraint('initial', 'v - v_0')
ocp.add_constraint('initial', 'gamma - gamma_0')
ocp.add_constraint('initial', 'psi - psi_0')

ocp.add_constraint('terminal', 'h - h_f')
ocp.add_constraint('terminal', 'v - v_f')
ocp.add_constraint('terminal', 'gamma - gamma_f')

ocp.add_inequality_constraint('path', 'alpha', lower_limit='alpha_min', upper_limit='alpha_max',
                              regularizer=PenaltyConstraintHandler('eps_alpha', method='sec'))
# ocp.add_inequality_constraint('path', 'beta', lower_limit='beta_min', upper_limit='beta_max',
#                               regularizer=PenaltyConstraintHandler('eps_beta', method='sec'))

sym_ocp = SymOCP(ocp)
comp_ocp = sym_ocp.compile()
comp_adj = SymAdjoints(sym_ocp).compile()
comp_dual = SymDual(ocp, control_method='differential').compile()
comp_bvp = convert_dual_to_bvp(comp_dual)

# guess_bvp = initialize_guess(comp_bvp)
# guess_ocp = initialize_guess(comp_ocp)
# guess_adj = initialize_guess(comp_adj)
# guess_dua = initialize_guess(comp_dual, t_span=[0, 3], x=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]),
#                              p=2, nu0=(1, 2, 3, 4, 5, 6, 7))

x_0 = np.array([260_000., 0., 0., 25_000., -1 / 180 * np.pi, np.pi/2])
lam_0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
u_0 = np.array([10 / 180 * np.pi, 0.])
guess_prop_bvp = propagate_guess(comp_bvp, initial_states=np.concatenate((x_0, lam_0, u_0)), t_span=100, reverse=True)
guess_prop_ocp = propagate_guess_ocp(comp_ocp, 100, x_0, (7.5*np.pi/180, 0))
guess_prop_ocp_fun = propagate_guess_ocp(comp_ocp, 100, x_0, lambda _t, _x, _p, _k: np.asarray([_t, _x[1]]))
guess_prop = propagate_guess_dual(comp_dual, 100, x_0, lam_0, (7.5*np.pi/180, 0))

# guess_constants_matched_bvp = match_constants_to_boundary_conditions(comp_bvp, guess_prop_bvp)
# guess_constants_matched_ocp = match_constants_to_boundary_conditions(comp_ocp, guess_prop_ocp)
guess_constants_matched = match_constants_to_boundary_conditions(comp_dual, guess_prop)

# guess_states_matched_bvp = match_states_to_boundary_conditions(comp_bvp, guess_prop_bvp)
# guess_states_matched_ocp = match_states_to_boundary_conditions(comp_ocp, guess_prop_ocp)
# guess_states_matched = match_states_to_boundary_conditions(comp_dual, guess_prop)

guess_adjoints_matched_mid = match_adjoints(comp_dual, guess_constants_matched, quadrature='midpoint')
guess_adjoints_matched_lin = match_adjoints(comp_dual, guess_constants_matched, quadrature='linear')
guess_adjoints_matched_sim = match_adjoints(comp_dual, guess_constants_matched, quadrature='simpson')

# sol_set = load_sol_set('sol_set.data')
# sol = sol_set[-1]
#
# x_dot = comp_dual.compute_dynamics(sol.t[0], sol.x[:, 0], sol.u[:, 0], sol.p, sol.k)
# bc = comp_dual.compute_boundary_conditions(sol.t, sol.x, sol.p, sol.k)
# cost = comp_dual.compute_cost(sol.t, sol.x, sol.u, sol.p, sol.k)
#
# lam_dot = comp_dual.compute_costate_dynamics(sol.t[0], sol.x[:, 0], sol.lam[:, 0], sol.u[:, 0], sol.p, sol.k)
# adj_bc = comp_dual.compute_adjoint_boundary_conditions(
#         sol.t, sol.x, sol.lam, sol.u, sol.p, np.concatenate((sol.nu0, sol.nuf)), sol.k)
# ham = comp_dual.compute_hamiltonian(sol.t[0], sol.x[:, 0], sol.lam[:, 0], sol.u[:, 0], sol.p, sol.k)
#
# comp_hand = comp_dual.control_handler
# u_dot = comp_hand.compute_control_dynamics(sol.t[0], sol.x[:, 0], sol.lam[:, 0], sol.u[:, 0], sol.p, sol.k)
# # dh_du = comp_hand.compute_control_boundary_conditions(sol.t[0], sol.x[:, 0], sol.lam[:, 0], sol.u[:, 0], sol.p, sol.k)
# dh_du = comp_dual.compute_control_law(sol.t[0], sol.x[:, 0], sol.lam[:, 0], sol.u[:, 0], sol.p, sol.k)
#
# comp_bvp = convert_dual_to_bvp(comp_dual)
#
# sol_bvp = comp_bvp.preprocess_data(sol)
# y_dot = comp_bvp.compute_dynamics(sol_bvp.t[0], sol_bvp.x[:, 0], sol_bvp.p, sol_bvp.k)
# dual_bc = comp_bvp.compute_boundary_conditions(sol_bvp.t, sol_bvp.x, sol_bvp.p, sol_bvp.k)
# sol_back = comp_bvp.post_process_data(sol_bvp)

# sp_bvp = SciPyBVP(comp_bvp)
# sp_tau, sp_x, sp_p = sp_bvp.preprocess(sol)
# scipy_sol = SciPySolver(comp_bvp, verbose=True).solve(sol_set[1].k, sol_set[0])

# with Timer(prefix='Compilation Time:'):
#     sym_ocp = SymOCP(ocp)
#     sym_dual = SymDual(sym_ocp)
#     sym_bvp = SymDualOCP(sym_ocp, sym_dual, control_method='differential')
#     comp_dual_ocp = CompDualOCP(sym_bvp)
#     num_solver = ScipySolveBVP(comp_dual_ocp, bc_tol=1e-8)
#
# guess = auto_propagate_guess(comp_dual_ocp, control=(20/180*3.14159, 0), t_span=100)
# seed_sol = num_solver.solve(guess.k, guess)
# sol_set = SolutionSet(sym_bvp, seed_sol)
#
# cont = ContinuationHandler(sol_set)
# cont.add_linear_series(100, {'h_f': 200_000, 'v_f': 10_000})
# cont.add_linear_series(50, {'h_f': 80_000, 'v_f': 2_500, 'gamma_f': -5 / 180 * 3.14159})
# cont.add_linear_series(90, {'xi': np.pi / 2}, bisection=True)
# sol_set = cont.run_continuation(num_solver)
#
# sol_set.save('sol_set.data')
