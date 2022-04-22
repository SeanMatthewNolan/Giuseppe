from typing import Union

from giuseppe.problems.bvp import CompBVP, AdiffBVP, BVPSol
from giuseppe.problems.dual import CompDual, CompDualOCP, AdiffDual, AdiffDualOCP, DualSol, DualOCPSol
from giuseppe.problems.ocp import CompOCP, AdiffOCP, OCPSol

AnyProblem = Union[CompBVP, CompOCP, CompDual, CompDualOCP,
                   AdiffBVP, AdiffOCP, AdiffDual, AdiffDualOCP]
AnySolution = Union[BVPSol, OCPSol, DualSol, DualOCPSol]
