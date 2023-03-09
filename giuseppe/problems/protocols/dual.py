from typing import Protocol, runtime_checkable, Union

from .ocp import OCP, VectorizedOCP
from .adjoints import Adjoints, VectorizedAdjoints
from .control_handlers import AlgebraicControlHandler, DifferentialControlHandler, VectorizedAlgebraicControlHandler,\
    VectorizedDifferentialControlHandler


@runtime_checkable
class Dual(OCP, Adjoints, Protocol):
    prob_class = 'dual'
    control_handler: Union[None, AlgebraicControlHandler, DifferentialControlHandler]


@runtime_checkable
class DualVectorized(Dual, VectorizedOCP, VectorizedAdjoints, Protocol):
    prob_class = 'dual'
    control_handler: Union[None, VectorizedAlgebraicControlHandler, VectorizedDifferentialControlHandler]
