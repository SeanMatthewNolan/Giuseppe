import os

import giuseppe

os.chdir(os.path.dirname(__file__))  # Set directory to current location

G = 5.3
NM2FT = 6076.1
T_GUESS = 2.5

lunar = giuseppe.problems.input.StrInputProb()

lunar.set_independent('t')

lunar.add_state('h', 'v_h')
lunar.add_state('x', 'v_x')
lunar.add_state('v_h', 'a * sin(β) - g')
lunar.add_state('v_x', 'a * cos(β)')

lunar.add_control('β')

lunar.add_constant('a', 3 * G)
lunar.add_constant('g', G)
lunar.add_constant('r_m', 938 * NM2FT)

lunar.add_constant('h_0', 0)
lunar.add_constant('x_0', 0)
lunar.add_constant('v_h_0', 0)
lunar.add_constant('v_x_0', 0)

lunar.add_constant('h_f', G * T_GUESS ** 2)
lunar.add_constant('v_h_f', G * T_GUESS)
lunar.add_constant('v_x_f', G * T_GUESS)

lunar.set_cost('0', '0', 't')

lunar.add_constraint('initial', 't')
lunar.add_constraint('initial', 'h - h_0')
lunar.add_constraint('initial', 'x - x_0')
lunar.add_constraint('initial', 'v_h - v_h_0')
lunar.add_constraint('initial', 'v_x - v_x_0')

lunar.add_constraint('terminal', 'h - h_f')
lunar.add_constraint('terminal', 'v_h - v_h_f')
lunar.add_constraint('terminal', 'v_x - v_x_f')

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_lunar = giuseppe.problems.symbolic.SymDual(lunar, control_method='algebraic')
    solver = giuseppe.numeric_solvers.SciPySolver(comp_lunar, use_jit_compile=False)

guess = giuseppe.guess_generation.auto_propagate_guess(comp_lunar, control=45 / 180 * 3.14159, t_span=T_GUESS)

cont = giuseppe.continuation.ContinuationHandler(solver, guess)
cont.add_linear_series(5, {'h_f': 50_000, 'v_h_f': 0, 'v_x_f': 5_780})
sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
