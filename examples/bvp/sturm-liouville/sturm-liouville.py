import os

import numpy as np

from giuseppe.continuation import ContinuationHandler
from giuseppe.guess_generation import initialize_guess
from giuseppe.numeric_solvers import SciPySolver
from giuseppe.problems.symbolic import SymBVP
from giuseppe.problems.input import StrInputProb
from giuseppe.utils import Timer

os.chdir(os.path.dirname(__file__))  # Set directory to file location

sturm_liouville = StrInputProb()

sturm_liouville.set_independent('x')

sturm_liouville.add_state('y', 'yp')
sturm_liouville.add_state('yp', '-k**2 * y')

sturm_liouville.add_parameter('k')

sturm_liouville.add_constant('x_0', 0.)
sturm_liouville.add_constant('x_f', 1.)
sturm_liouville.add_constant('y_0', 0.)
sturm_liouville.add_constant('y_f', 0.)
sturm_liouville.add_constant('a', 1.)

sturm_liouville.add_constraint('initial', 'x - x_0')
sturm_liouville.add_constraint('initial', 'y - y_0')
sturm_liouville.add_constraint('initial', 'yp - a * k')

sturm_liouville.add_constraint('terminal', 'x - x_f')
sturm_liouville.add_constraint('terminal', 'y - y_f')

with Timer(prefix='Compilation Time:'):
    comp_bvp = SymBVP(sturm_liouville)
    solver = SciPySolver(comp_bvp)

guess = initialize_guess(comp_bvp, t_span=np.linspace(0, 1, 3))

cont = ContinuationHandler(solver, guess)
cont.add_linear_series(10, {'a': 100})
sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
