from typing import Union, Tuple, Optional

from giuseppe.problems.bvp import CompBVP
from giuseppe.problems.dual import CompDual, CompDualOCP
from giuseppe.problems.ocp import CompOCP


def sift_ocp_and_dual(comp_prob: Union[CompBVP, CompOCP, CompDual, CompDualOCP]) \
        -> Tuple[Optional[Union[CompOCP, CompBVP]], Optional[CompDual]]:
    if isinstance(comp_prob, CompDualOCP):
        prob = comp_prob.comp_ocp
        dual = comp_prob.comp_dual
    elif isinstance(comp_prob, CompBVP) or isinstance(comp_prob, CompOCP):
        prob = comp_prob
        dual = None
    elif isinstance(comp_prob, CompDual):
        prob = None
        dual = comp_prob
    else:
        raise TypeError(f'{type(comp_prob)} not supported')

    return prob, dual
