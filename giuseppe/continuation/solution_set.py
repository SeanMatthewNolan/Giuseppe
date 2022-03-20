from abc import abstractmethod
from collections.abc import Iterable, MutableSequence, Hashable
from typing import Union, overload

from .solution_subset import SolutionSubset
from ..problems.bvp import SymBVP, BVPSol
from ..utils.mixins import Picky


# TODO: add annotations to solution set
class SolutionSet(MutableSequence, Picky):

    SUPPORTED_INPUTS = Union[SymBVP]

    def __init__(self, problem: SymBVP, seed_solution: BVPSol):
        Picky.__init__(self, problem)

        self.problem: SymBVP = problem
        self.seed_solution: BVPSol = seed_solution

        self.subsets: list[SolutionSubset] = [
            SolutionSubset(data=[self.seed_solution], description='the seed solution (guess)')]

        # Annotations
        self.constant_names: tuple[Hashable, ...] = tuple(str(constant) for constant in self.problem.constants)

    def get_last(self):
        assert len(self.subsets) > 0, 'Solution set is empty! It should be intialized with seed solution/guess'
        if len(self.subsets[-1]) > 0:
            return self.subsets[-1][-1]
        else:
            assert len(self.subsets[-2]) > 0, 'The last two solutions subsets are empty!'
            return self.subsets[-2][-1]

    def insert(self, index: int, sol_subset: SolutionSubset) -> None:
        self.subsets.insert(index, sol_subset)

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> SolutionSubset: ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[SolutionSubset]: ...

    def __getitem__(self, i: int) -> SolutionSubset:
        return self.subsets.__getitem__(i)

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: SolutionSubset) -> None: ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[SolutionSubset]) -> None: ...

    def __setitem__(self, i: int, o: SolutionSubset) -> None:
        self.__setitem__(i, o)

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None: ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None: ...

    def __delitem__(self, i: int) -> None:
        self.subsets.__delitem__(i)

    def __len__(self) -> int:
        return self.subsets.__len__()
