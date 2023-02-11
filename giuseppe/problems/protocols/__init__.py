from typing import Union

from .bvp import BVP
from .ocp import OCP
from .adjoints import Adjoints
from .control_handlers import AlgebraicControlHandler, DifferentialControlHandler
from .dual import Dual

Problem = Union[BVP, OCP, Adjoints, Dual]
