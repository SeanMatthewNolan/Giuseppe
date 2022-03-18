from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from .symbolic import SymDualOCP
from ..utils.typing import NumbaFloat, NumbaArray
from ..utils.complilation import lambdify, jit_compile
from ..utils.mixins import Picky


class CompDualOCP(Picky):
    SUPPORTED_INPUTS: type = Union[SymDualOCP]

    def __init__(self, source_ocp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_ocp)

        self.source_ocp = deepcopy(source_ocp)  # source ocp is copied here for reference as it may be mutated later
