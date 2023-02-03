from copy import deepcopy
from typing import Optional, Tuple, Union

from .ocp import SymOCP, CompOCP
from .dual import SymDual, CompDual
from .control_handlers import ImplicitAlgebraicControlHandler, ExplicitAlgebraicControlHandler, \
    DifferentialControlHandler, CompImplicitAlgebraicControlHandler, CompExplicitAlgebraicControlHandler, \
    CompDifferentialControlHandler


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
            self.control_handler = ExplicitAlgebraicControlHandler(primal, dual)
        elif self.control_method.lower() == 'differential':
            self.control_handler = DifferentialControlHandler(primal, dual)
        elif self.control_method.lower() == 'implicit':
            self.control_handler = ImplicitAlgebraicControlHandler(primal, dual)
        else:
            raise NotImplementedError(
                    f'\"{control_method}\" is not an implemented control method. Try \"differential\".')

    def compile(self, use_jit_compile: bool = True, cost_quadrature: str = 'simpson')\
            -> Tuple[CompOCP, CompDual, Union[CompImplicitAlgebraicControlHandler,
                     CompExplicitAlgebraicControlHandler, CompDifferentialControlHandler]]:

        comp_primal = CompOCP(self.primal, use_jit_compile=use_jit_compile, cost_quadrature=cost_quadrature)
        comp_dual = CompDual(self.dual, use_jit_compile=use_jit_compile)
        comp_control_handler = self.control_handler.compile(comp_dual, use_jit_compile=use_jit_compile)

        return comp_primal, comp_dual, comp_control_handler
