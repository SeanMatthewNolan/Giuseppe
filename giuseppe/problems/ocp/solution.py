from dataclasses import dataclass
from typing import Optional

from giuseppe.problems.bvp.solution import BVPSol
from giuseppe.utils.typing import NPArray, EMPTY_2D_ARRAY


@dataclass
class OCPSol(BVPSol):
    u: NPArray = EMPTY_2D_ARRAY

    cost: Optional[float] = None
