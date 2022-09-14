from typing import Union, Optional, Protocol

from giuseppe.io import Solution, SolutionSet


class ContinuationHandler(Protocol):
    def __init__(self, *args, **kwargs):
        self.solution_set: Optional[SolutionSet] = None

    def run(self, num_solver, seed: Union[Solution, SolutionSet]) -> SolutionSet:
        if isinstance(seed, Solution):
            self.solution_set = SolutionSet()
