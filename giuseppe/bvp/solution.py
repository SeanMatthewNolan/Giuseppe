from dataclasses import dataclass
from typing import Optional

from ..utils.typing import NPArray, EMPTY_ARRAY, EMPTY_2D_ARRAY


@dataclass
class BVPSol:
    t: NPArray = EMPTY_ARRAY
    x: NPArray = EMPTY_2D_ARRAY
    p: NPArray = EMPTY_ARRAY
    k: NPArray = EMPTY_ARRAY

    aux: Optional[dict] = None
