from abc import abstractmethod
from typing import MutableSequence, Optional, Iterable, overload

from ..problems.bvp import BVPSol


class SolutionSubset(MutableSequence):
    def __init__(self, data: Optional[Iterable[BVPSol]] = None, description: str = ''):
        if data is None:
            data = []
        else:
            data = list(data)

        self.data: list[BVPSol] = data
        self.description = description

    def insert(self, index: int, solution: BVPSol) -> None:
        self.data.insert(index, solution)

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> BVPSol: ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[BVPSol]: ...

    def __getitem__(self, i: int) -> BVPSol:
        return self.data.__getitem__(i)

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: BVPSol) -> None: ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[BVPSol]) -> None: ...

    def __setitem__(self, i: int, o: BVPSol) -> None:
        self.data.__setitem__(i, o)

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None: ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None: ...

    def __delitem__(self, i: int) -> None:
        self.data.__delitem__(i)

    def __len__(self) -> int:
        return self.data.__len__()
