from giuseppe.io import InputOCP
from giuseppe.problems.dual import SymDual, SymDualOCP
from giuseppe.problems.ocp import SymOCP

ocp = InputOCP()

ocp.set_independent('t')

ocp.add_state('x', 'v*cos(theta)')
ocp.add_state('y', 'v*sin(theta)')
ocp.add_state('v', '-g*sin(theta)')

ocp.add_control('theta')

ocp.add_constant('g', 32.2)

ocp.add_constant('x_0')
ocp.add_constant('y_0')
ocp.add_constant('v_0')

ocp.add_constant('x_f')
ocp.add_constant('y_f')

ocp.set_cost('0', '1', '0')

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'x - x_0')
ocp.add_constraint('initial', 'y - y_0')
ocp.add_constraint('initial', 'v - v_0')

ocp.add_constraint('terminal', 'x - x_f')
ocp.add_constraint('terminal', 'y - y_f')

sym_ocp = SymOCP(ocp)
sym_dual = SymDual(sym_ocp)
sym_bvp_alg = SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
sym_bvp_dif = SymDualOCP(sym_ocp, sym_dual, control_method='differential')
