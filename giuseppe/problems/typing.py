from typing import Union

from giuseppe.problems.bvp import CompBVP, BVPSol
from giuseppe.problems.dual import CompDual, CompDualOCP, DualSol, DualOCPSol
from giuseppe.problems.ocp import CompOCP, OCPSol

AnyProblem = Union[CompBVP, CompOCP, CompDual, CompDualOCP]
AnySolution = Union[BVPSol, OCPSol, DualSol, DualOCPSol]
