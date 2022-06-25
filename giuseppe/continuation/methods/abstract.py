from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Optional

from giuseppe.problems.bvp import BVPSol
from ..solution_set import SolutionSet
from ...utils.typing import NPArray
from ...utils.strings import justify_str


class ContinuationSeries(Iterable, ABC):
    def __init__(self, solution_set: SolutionSet):
        self.solution_set: SolutionSet = solution_set

        self.current_step: int = 0
        self.num_completed_steps: int = 0
        self.estimated_num_steps: int = 0

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[NPArray, BVPSol]]:
        pass

    def __repr__(self):
        return 'Continuation Series'

    def __format__(self, format_spec: Optional[str]):
        raw_str = repr(self)

        if format_spec is None:
            return raw_str
        elif format_spec.isnumeric():
            out_len = int(format_spec)
            return justify_str(raw_str, out_len)
        else:
            raise ValueError(f'Unknown format code \'{format_spec}\' for object of type \'{type(self)}\'')
