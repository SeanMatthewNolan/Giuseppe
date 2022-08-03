from dataclasses import dataclass
from typing import Optional

from giuseppe.utils.typing import NPArray


@dataclass
class Solution:
    t: Optional[NPArray] = None
    x: Optional[NPArray] = None
    p: Optional[NPArray] = None
    k: Optional[NPArray] = None

    u: Optional[NPArray] = None

    lam: Optional[NPArray] = None
    nu0: Optional[NPArray] = None
    nuf: Optional[NPArray] = None

    aux: Optional[dict] = None

    converged: bool = False
