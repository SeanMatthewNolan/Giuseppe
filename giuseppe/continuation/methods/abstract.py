from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from giuseppe.problems.bvp import BVPSol
from ..solution_set import SolutionSet
from ...utils.typing import NPArray


class ContinuationSeries(Iterable, ABC):
    def __init__(self, solution_set: SolutionSet):
        self.solution_set: SolutionSet = solution_set

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[NPArray, BVPSol]]:
        pass
