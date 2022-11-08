import giuseppe

ocp = giuseppe.io.InputOCP()

ocp.set_independent('tau')

ocp.add_expression('eta', 'v_0 * t_f / r_0')
ocp.add_expression('T_bar', 'T * t_f / v_0')

ocp.add_state('r_bar', 'u_bar * eta')
ocp.add_state('theta', 'v_bar / r_bar * eta')
ocp.add_state('u_bar', '(v_bar**2 / r_bar - 1 / r_bar ** 2) * eta + T_bar * sin(alpha) / (m_0 - m_dot * tau * t_f)')
ocp.add_state('v_bar', '-u_bar * v_bar / r_bar + T_bar * cos(alpha) / (m_0 - m_dot * tau * t_f)')

ocp.add_control('alpha')

T = 20
ISP = 6000
RE = 6378e3
G0 = 9.80665
MU = 3.986004418e14
H0 = 300e3

ocp.add_constant('mu', MU)
ocp.add_constant('T', T)
ocp.add_constant('m_dot', T/ISP/G0)
ocp.add_constant('m_0', 1500)

ocp.add_constant('r_0', RE + H0)
ocp.add_constant('u_0', 0)
ocp.add_constant('v_0', (MU / (RE + H0))**0.5)
ocp.add_constant('theta_0', 0)

ocp.add_constant('u_f', 0)
ocp.add_constant('t_f', 24 * 3600 * .1)

ocp.add_constant('eps', 1)

ocp.set_cost('0', '0', '-r_bar')

ocp.add_constraint('initial', 'tau')
ocp.add_constraint('initial', 'r_bar - 1')
ocp.add_constraint('initial', 'u_bar')
ocp.add_constraint('initial', 'v_bar - 1')
ocp.add_constraint('initial', 'theta')

ocp.add_constraint('terminal', 'tau - 1')
ocp.add_constraint('terminal', 'u_bar')

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    sym_ocp = giuseppe.problems.SymOCP(ocp)
    sym_dual = giuseppe.problems.SymDual(sym_ocp)
    sym_bvp = giuseppe.problems.SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
    comp_dual_ocp = giuseppe.problems.CompDualOCP(sym_bvp)
    num_solver = giuseppe.numeric_solvers.ScipySolveBVP(comp_dual_ocp)

guess = giuseppe.guess_generators.auto_propagate_guess(comp_dual_ocp, control=0.0, t_span=1)
seed_sol = num_solver.solve(guess.k, guess)
seed_sol.save('seed.data')

sol_set = giuseppe.io.SolutionSet(sym_ocp, seed_sol)
cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(100, {'t_f': 24 * 3600 * 1}, bisection=True)
cont.add_linear_series(100, {'t_f': 24 * 3600 * 10}, bisection=True)
sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')
