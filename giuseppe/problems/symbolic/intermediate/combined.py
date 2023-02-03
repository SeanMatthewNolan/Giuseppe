from typing import Optional
from copy import deepcopy

from giuseppe.problems.symbolic.intermediate.control_handlers import ImplicitAlgebraicControlHandler,\
    ExplicitAlgebraicControlHandler, DifferentialControlHandler
from giuseppe.problems.symbolic.intermediate.ocp import SymOCP
from giuseppe.problems.symbolic.intermediate.dual import SymDual


class SymCombined:
    def __init__(self, primal: SymOCP, dual: Optional[SymDual] = None, control_method: Optional[str] = 'differential'):

        self.primal: SymOCP = deepcopy(primal)

        if dual is None:
            self.dual: SymDual = SymDual(primal)
        else:
            self.dual: SymDual = deepcopy(dual)

        self.control_method: Optional[str] = control_method
        if self.control_method is None:
            self.control_handler = None
        elif self.control_method.lower() == 'algebraic':
            self.control_handler = AlgebraicControlHandler(primal, dual)
        elif self.control_method.lower() == 'differential':
            self.control_handler = DifferentialControlHandler(primal, dual)
        else:
            raise NotImplementedError(
                    f'\"{control_method}\" is not an implemented control method. Try \"differential\".')
