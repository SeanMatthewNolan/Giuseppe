from __future__ import annotations

from typing import Protocol

from giuseppe.data_classes import Solution
from giuseppe.numeric_solvers.scipy.solver import _scipy_bvp_sol
from giuseppe.utils.typing import NPArray


class SciPyBVP(Protocol):
    def dynamics(self, tau_vec: NPArray, x_vec: NPArray, p: NPArray, k: NPArray) -> NPArray:
        ...

    def boundary_conditions(self, x0: NPArray, xf: NPArray, p: NPArray, k: NPArray):
        ...

    def preprocess(self, guess: Solution) -> tuple[NPArray, NPArray, NPArray]:
        ...

    def postprocess(self, scipy_sol: _scipy_bvp_sol) -> Solution:
        ...
