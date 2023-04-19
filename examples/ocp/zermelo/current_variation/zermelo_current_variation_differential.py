import giuseppe

zermelo = giuseppe.problems.input.StrInputProb()

zermelo.set_independent('t')

zermelo.add_expression('current', 'c*y')

zermelo.add_state('x', 'v*cos(theta) + current')
zermelo.add_state('y', 'v*sin(theta)')

zermelo.add_control('theta')

zermelo.add_constant('v', 1)
zermelo.add_constant('c', 0)

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

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_prob = giuseppe.problems.symbolic.SymDual(zermelo, control_method='differential')
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_prob)

guess = giuseppe.guess_generation.auto_propagate_guess(comp_prob, control=0, t_span=1)

cont = giuseppe.continuation.ContinuationHandler(num_solver, guess)
cont.add_linear_series(10, {'x_f': 0, 'y_f': 0})
cont.add_linear_series(6, {'c': -1})
cont.add_linear_series(6, {'c': 1})
sol_set = cont.run_continuation()

sol_set.save('current_variation_sol_set.data')
