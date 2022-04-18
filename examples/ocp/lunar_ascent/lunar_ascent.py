import pickle

import numpy as np

import giuseppe

G = 5.3
NM2FT = 6076.1
T_GUESS = 2.5

lunar = giuseppe.io.InputOCP()

lunar.set_independent('t')

lunar.add_state('h', 'v_h')
lunar.add_state('θ', 'v_θ / (h + r_m)')
lunar.add_state('v_h', 'a * sin(β) - g')
lunar.add_state('v_θ', 'a * cos(β)')

lunar.add_control('β')

lunar.add_constant('a', 3 * G)
lunar.add_constant('g', G)
lunar.add_constant('r_m', 938 * NM2FT)

lunar.add_constant('h_0', 0)
lunar.add_constant('θ_0', 0)
lunar.add_constant('v_h_0', 0)
lunar.add_constant('v_θ_0', 0)

lunar.add_constant('h_f', G * T_GUESS ** 2)
lunar.add_constant('v_h_f', G * T_GUESS)

lunar.set_cost('0', '0', 't')

lunar.add_constraint('initial', 't')
lunar.add_constraint('initial', 'h - h_0')
lunar.add_constraint('initial', 'θ - θ_0')
lunar.add_constraint('initial', 'v_h - v_h_0')
lunar.add_constraint('initial', 'v_θ - v_θ_0')

lunar.add_constraint('terminal', 'h - h_f')
lunar.add_constraint('terminal', 'v_h - v_h_f')

with giuseppe.utils.Timer(prefix='Complilation Time:'):
    sym_ocp = giuseppe.problems.SymOCP(lunar)
    sym_dual = giuseppe.problems.SymDual(sym_ocp)
    sym_bvp = giuseppe.problems.SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
    comp_dual_ocp = giuseppe.problems.CompDualOCP(sym_bvp, use_jit_compile=False)
    num_solver = giuseppe.numeric_solvers.ScipySolveBVP(comp_dual_ocp, use_jit_compile=False)

guess = giuseppe.guess_generators.auto_linear_guess(comp_dual_ocp, t_span=T_GUESS)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = giuseppe.continuation.SolutionSet(sym_bvp, seed_sol)
cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(1000, {'h_f': 1_000}, bisection=True)

with giuseppe.utils.Timer(prefix='Continuation Time:'):
    for series in cont.continuation_series:
        for k, last_sol in series:
            sol_i = num_solver.solve(k, last_sol)
            sol_set.append(sol_i)

with open('sol_set.data', 'wb') as file:
    pickle.dump(sol_set, file)
