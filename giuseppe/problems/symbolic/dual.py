from typing import Optional

from giuseppe.problems.protocols import Dual
from giuseppe.data_classes.annotations import Annotations

from .input import StrInputProb
from .adjoints import SymAdjoints
from .ocp import SymOCP


class SymDual(SymOCP, SymAdjoints, Dual):
    def __init__(self, input_data: StrInputProb,
                 control_method: Optional[str] = 'differential', use_jit_compile: bool = True):

        super().__init__(input_data, use_jit_compile=use_jit_compile)
        super()._sympify_adjoint_information(self)
        super()._compile_adjoint_information(self)

        self.prob_class = 'dual'

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
