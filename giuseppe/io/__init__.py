from .solution import Solution, load as load_sol
from .solution_set import SolutionSet

from ..problems.input import AdiffInputProb
from ..problems.bvp import InputBVP
from ..problems.ocp import InputOCP

StrInputBVP = InputBVP
StrInputOCP = InputOCP
AdiffInputBVP = AdiffInputProb
AdiffInputOCP = AdiffInputProb
