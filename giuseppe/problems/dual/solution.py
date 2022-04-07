from dataclasses import dataclass
from typing import Optional

from giuseppe.problems.bvp.solution import BVPSol
from giuseppe.problems.ocp.solution import OCPSol
from giuseppe.utils.typing import NPArray, EMPTY_ARRAY, EMPTY_2D_ARRAY, check_if_any_exist, sift_nones_from_dict


@dataclass
class DualSol:
    lam: NPArray = EMPTY_2D_ARRAY
    nu0: NPArray = EMPTY_ARRAY
    nuf: NPArray = EMPTY_ARRAY


@dataclass
class DualOCPSol(DualSol, OCPSol):
    pass


# TODO Add dimensional consistency checks
# TODO Look at use of overload to take in already form solutions and generic dict_
class Solution:
    def __new__(
            cls, t: Optional[NPArray] = None, x: Optional[NPArray] = None, lam: Optional[NPArray] = None,
            u: Optional[NPArray] = None, p: Optional[NPArray] = None, nu0: Optional[NPArray] = None,
            nuf: Optional[NPArray] = None, k: Optional[NPArray] = None, cost: Optional[float] = None,
            aux: Optional[dict] = None, converged: Optional[bool] = None):

        is_bvp: bool = check_if_any_exist(t, x, p, k, aux)
        is_ocp: bool = check_if_any_exist(u, cost)
        is_dual: bool = check_if_any_exist(lam, nu0, nuf)

        kwargs = sift_nones_from_dict(
                {'t': t, 'x': x, 'lam': lam, 'u': u, 'p': p, 'nu0': nu0, 'nuf': nuf, 'k': k,
                 'cost': cost, 'aux': aux, 'converged': converged}
        )

        if is_dual and (is_bvp or is_ocp):
            return DualOCPSol(**kwargs)
        elif is_dual:
            return DualSol(**kwargs)
        elif is_ocp:
            return OCPSol(**kwargs)
        elif is_bvp:
            return BVPSol(**kwargs)
        else:
            return DualOCPSol(**kwargs)
