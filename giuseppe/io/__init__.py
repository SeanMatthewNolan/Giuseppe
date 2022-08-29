from .solution import Solution, load as load_sol
from .solution_set import SolutionSet

from ..problems.bvp import InputBVP, AdiffInputBVP
from ..problems.ocp import InputOCP, AdiffInputOCP

StrInputBVP = InputBVP
StrInputOCP = InputOCP
AdiffInputBVP = AdiffInputBVP
AdiffInputOCP = AdiffInputOCP
