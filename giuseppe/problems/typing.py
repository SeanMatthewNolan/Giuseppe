from typing import Union

from giuseppe.problems.bvp import CompBVP, AdiffBVP
from giuseppe.problems.dual import CompDual, CompDualOCP, AdiffDual, AdiffDualOCP
from giuseppe.problems.ocp import CompOCP, AdiffOCP

AnyProblem = Union[CompBVP, CompOCP, CompDual, CompDualOCP,
                   AdiffBVP, AdiffOCP, AdiffDual, AdiffDualOCP]
