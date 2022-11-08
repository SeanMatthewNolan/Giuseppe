import os; os.chdir(os.path.dirname(__file__))

import giuseppe

ocp = giuseppe.io.InputOCP()

ocp.set_independent('t')

ocp.add_state('r', 'u')
ocp.add_state('theta', 'v / r')  # theta is reducible
ocp.add_state('u', 'v ** 2 / r - mu / r ** 2 + T * sin(phi) / (m_0 - m_dot * t)')
ocp.add_state('v', '-u * v / r + T * cos(phi) / (m_0 - m_dot * t)')

ocp.add_control('phi')

T = 20
ISP = 6000
RE = 6378e3
G0 = 9.80665
MU = 3.986004418e14
H0 = 300e3
M0 = 1500

ocp.add_constant('mu', MU)
ocp.add_constant('T', T)
ocp.add_constant('m_dot', T / ISP / G0)
ocp.add_constant('m_0', M0)

ocp.add_constant('r_0', RE + H0)
ocp.add_constant('u_0', 0)
# ocp.add_constant('v_0', (MU / (RE + H0))**0.5)
ocp.add_constant('theta_0', 0)

ocp.add_constant('u_f', 0)
ocp.add_constant('t_f', 24 * 3600 * .1)

ocp.set_cost('0', '0', '-r')

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'r - r_0')
ocp.add_constraint('initial', 'u - u_0')
ocp.add_constraint('initial', 'v - sqrt(mu / r_0)')
ocp.add_constraint('initial', 'theta - theta_0')

ocp.add_constraint('terminal', 't - t_f')
ocp.add_constraint('terminal', 'u - u_f')

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    sym_ocp = giuseppe.problems.SymOCP(ocp)
    sym_dual = giuseppe.problems.SymDual(sym_ocp)
    sym_bvp = giuseppe.problems.SymDualOCP(sym_ocp, sym_dual, control_method='differential')
    comp_dual_ocp = giuseppe.problems.CompDualOCP(sym_bvp)
    num_solver = giuseppe.numeric_solvers.ScipySolveBVP(comp_dual_ocp)

guess = giuseppe.guess_generators.auto_propagate_guess(comp_dual_ocp, control=0.0, t_span=24 * 3600 * 0.1)
seed_sol = num_solver.solve(guess.k, guess)
seed_sol.save('seed.data')

sol_set = giuseppe.io.SolutionSet(sym_ocp, seed_sol)
cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(100, {'t_f': 24 * 3600 * 0.5}, bisection=True)
cont.add_linear_series(100, {'t_f': 24 * 3600 * 4}, bisection=True)
sol_set = cont.run_continuation(num_solver)

sol_set.save('sol_set.data')
