import giuseppe

zermelo = giuseppe.io.InputOCP()

zermelo.set_independent('t')

zermelo.add_expression('current', 'c*y')

zermelo.add_state('x', 'v*cos(theta) + current')
zermelo.add_state('y', 'v*sin(theta)')

zermelo.add_control('theta')

zermelo.add_constant('v', 1)
zermelo.add_constant('c', -1)

zermelo.add_constant('x_0', 3.5)
zermelo.add_constant('y_0', -1.8)

zermelo.add_constant('x_f', 0.)
zermelo.add_constant('y_f', 0.)

zermelo.set_cost('0', '0', 't')

zermelo.add_constraint('initial', 't')
zermelo.add_constraint('initial', 'x - x_0')
zermelo.add_constraint('initial', 'y - y_0')

zermelo.add_constraint('terminal', 'x - x_f')
zermelo.add_constraint('terminal', 'y - y_f')

with giuseppe.utils.Timer(prefix='Complilation Time:'):
    sym_ocp = giuseppe.problems.SymOCP(zermelo)
    sym_dual = giuseppe.problems.SymDual(sym_ocp)
    sym_bvp = giuseppe.problems.SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
    comp_dual_ocp = giuseppe.problems.CompDualOCP(sym_bvp, use_jit_compile=False)
    num_solver = giuseppe.numeric_solvers.ScipySolveBVP(comp_dual_ocp, use_jit_compile=False)

guess = giuseppe.guess_generators.generate_constant_guess(comp_dual_ocp)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = giuseppe.io.SolutionSet(sym_bvp, seed_sol)
cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(6, {'c': 1})

cont.run_continuation(num_solver)

sol_set.save('current_variation_sol_set.data')
