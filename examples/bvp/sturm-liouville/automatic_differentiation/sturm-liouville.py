import os

import casadi as ca
import numpy as np

from giuseppe.continuation import ContinuationHandler
from giuseppe.guess_generation import initialize_guess
from giuseppe.numeric_solvers import SciPySolver
from giuseppe.problems.automatic_differentiation import ADiffInputProb
from giuseppe.problems.automatic_differentiation import ADiffBVP

os.chdir(os.path.dirname(__file__))  # Set directory to file location

sturm_liouville = ADiffInputProb()

# Independent Variable
x = ca.SX.sym('x', 1)
sturm_liouville.set_independent(x)

# Constants
x_0 = ca.SX.sym('x_0', 1)
x_f = ca.SX.sym('x_f', 1)
y_0 = ca.SX.sym('y_0', 1)
y_f = ca.SX.sym('y_f', 1)
a = ca.SX.sym('a', 1)

sturm_liouville.add_constant(x_0, -1.)
sturm_liouville.add_constant(x_f, 1.)
sturm_liouville.add_constant(y_0, 0.)
sturm_liouville.add_constant(y_f, 0.)
sturm_liouville.add_constant(a, 1.)

# State Variables
y = ca.SX.sym()
yp = ca.SX.sym()

# Parameters
k = ca.SX.sym()

sturm_liouville.add_state(y, yp)
sturm_liouville.add_state(yp, -k**2 * y)

sturm_liouville.add_parameter(k)

sturm_liouville.add_constraint('initial', x - x_0)
sturm_liouville.add_constraint('initial', y - y_0)
sturm_liouville.add_constraint('initial', yp - a * k)

sturm_liouville.add_constraint('terminal', x - x_f)
sturm_liouville.add_constraint('terminal', y - y_f)

ad_bvp = ADiffBVP(sturm_liouville)

solver = SciPySolver(ad_bvp)

# guess = initialize_guess(ad_bvp, t_span=np.linspace(0, 1, 3))
#
# cont = ContinuationHandler(solver, guess)
# cont.add_linear_series(10, {'a': 100})
# sol_set = cont.run_continuation()
#
# sol_set.save('sol_set.data')
