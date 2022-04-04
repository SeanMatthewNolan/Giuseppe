from dataclasses import dataclass

from giuseppe.problems.ocp.solution import OCPSol
from giuseppe.utils.typing import NPArray, EMPTY_ARRAY, EMPTY_2D_ARRAY


@dataclass
class DualSol(OCPSol):
    lam: NPArray = EMPTY_2D_ARRAY
    nu0: NPArray = EMPTY_ARRAY
    nuf: NPArray = EMPTY_ARRAY
