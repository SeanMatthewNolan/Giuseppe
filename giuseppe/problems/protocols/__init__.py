from typing import Union

from .bvp import BVP, VectorizedBVP
from .ocp import OCP, VectorizedOCP
from .adjoints import Adjoints, VectorizedAdjoints
from .control_handlers import AlgebraicControlHandler, DifferentialControlHandler
from .regularizer import Regularizer
from .dual import Dual, VectorizedDual

Problem = Union[BVP, OCP, Adjoints, Dual]
