from typing import Union, Tuple, Optional

from giuseppe.problems.bvp import CompBVP, AdiffBVP
from giuseppe.problems.dual import CompDual, CompDualOCP, AdiffDual, AdiffDualOCP
from giuseppe.problems.ocp import CompOCP, AdiffOCP
from giuseppe.problems.typing import AnyProblem


def sift_ocp_and_dual(source_prob: AnyProblem) \
        -> Tuple[Optional[Union[CompOCP, CompBVP, AdiffOCP, AdiffBVP]], Optional[Union[CompDual, AdiffDual]]]:
    if isinstance(source_prob, CompDualOCP):
        prob = source_prob.comp_ocp
        dual = source_prob.comp_dual
    elif isinstance(source_prob, CompBVP) or isinstance(source_prob, CompOCP):
        prob = source_prob
        dual = None
    elif isinstance(source_prob, CompDual):
        prob = None
        dual = source_prob
    elif isinstance(source_prob, AdiffDualOCP):
        prob = source_prob.ocp
        dual = source_prob.dual
    elif isinstance(source_prob, AdiffBVP) or isinstance(source_prob, AdiffOCP):
        prob = source_prob
        dual = None
    elif isinstance(source_prob, AdiffDual):
        prob = None
        dual = source_prob
    else:
        raise TypeError(f'{type(source_prob)} not supported')

    return prob, dual
