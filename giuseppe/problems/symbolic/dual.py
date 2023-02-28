from typing import Union, Optional

from giuseppe.problems.input import StrInputProb
from giuseppe.problems.protocols import Dual
from giuseppe.data_classes.annotations import Annotations

from .adjoints import SymAdjoints, CompAdjoints
from .ocp import SymOCP, CompOCP


class SymDual(SymOCP, SymAdjoints):
    def __init__(self, input_data: Optional[StrInputProb] = None,
                 control_method: Optional[str] = 'differential'):

        super().__init__(input_data=input_data)
        super()._sympify_adjoint_information(self)

        self.control_method: Optional[str] = control_method
        if self.control_method is None:
            self.control_handler = None
        elif self.control_method.lower() == 'algebraic':
            from .control_handlers import SymAlgebraicControlHandler
            self.control_handler: SymAlgebraicControlHandler = SymAlgebraicControlHandler(self)
        elif self.control_method.lower() == 'differential':
            from .control_handlers import SymDifferentialControlHandler
            self.control_handler: SymDifferentialControlHandler = SymDifferentialControlHandler(self)
        else:
            raise NotImplementedError(
                    f'\"{control_method}\" is not an implemented control method. Try \"differential\".')

    def compile(self, use_jit_compile: bool = True, cost_quadrature: str = 'simpson') -> 'CompDual':
        return CompDual(self, use_jit_compile=use_jit_compile, cost_quadrature=cost_quadrature)


class CompDual(CompOCP, CompAdjoints, Dual):
    def __init__(self, source_dual: SymDual, use_jit_compile: bool = True, cost_quadrature: str = 'simpson'):
        self.source_dual = source_dual
        super().__init__(source_dual, use_jit_compile=use_jit_compile, cost_quadrature=cost_quadrature)
        del self.source_ocp
        super()._compile_adjoint_information(self.source_dual, self.source_dual)

        if self.source_dual.control_handler is not None:
            self.control_handler = self.source_dual.control_handler.compile(self, use_jit_compile=use_jit_compile)
        else:
            self.control_handler = None

        self.annotations: Annotations = self.source_dual.annotations
