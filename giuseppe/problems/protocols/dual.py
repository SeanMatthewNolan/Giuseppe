from typing import Protocol, runtime_checkable, Union

from .ocp import OCP
from .adjoints import Adjoints
from .control_handlers import AlgebraicControlHandler, DifferentialControlHandler


@runtime_checkable
class Dual(OCP, Adjoints, Protocol):
    control_handler: Union[None, AlgebraicControlHandler, DifferentialControlHandler]
