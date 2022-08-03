import warnings
from abc import abstractmethod
from collections.abc import Iterable, MutableSequence, Hashable
from copy import deepcopy
from typing import Union, overload

from giuseppe.problems.bvp import SymBVP, AdiffBVP
from giuseppe.problems.dual import SymDualOCP, AdiffDual, AdiffDualOCP
from giuseppe.problems.ocp import SymOCP, AdiffOCP
from giuseppe.utils.mixins import Picky
from .solution import Solution


# TODO: add annotations to solution set
class SolutionSet(MutableSequence, Picky):
    SUPPORTED_INPUTS = Union[SymBVP, SymOCP, SymDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP]

    def __init__(self, problem: SUPPORTED_INPUTS, seed_solution: Solution):
        Picky.__init__(self, problem)

        problem = deepcopy(problem)
        if type(problem) is SymDualOCP:
            self.constants = problem.ocp.constants
        elif isinstance(problem, AdiffDualOCP):
            self.constants = problem.dual.adiff_ocp.constants
        elif isinstance(problem, AdiffDual):
            self.constants = problem.adiff_ocp.constants
        elif isinstance(problem, AdiffOCP):
            self.constants = problem.constants
        else:
            self.constants = problem.constants

        if not seed_solution.converged:
            warnings.warn(
                'Seed solution is not converged! It is suggested to solve seed prior to initialization of solution set.'
            )

        self.seed_solution: Solution = seed_solution

        self.solutions: list[Solution] = [seed_solution]
        self.continuation_slices: list[slice] = []
        self.damned_sols: list[Solution] = []

        # Annotations
        self.constant_names: tuple[Hashable, ...] = tuple(str(constant) for constant in self.constants)

    def insert(self, index: int, solution: Solution) -> None:
        self.solutions.insert(index, solution)

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> Solution:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[Solution]:
        ...

    def __getitem__(self, i: int) -> Solution:
        return self.solutions.__getitem__(i)

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: Solution) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[Solution]) -> None:
        ...

    def __setitem__(self, i: int, o: Solution) -> None:
        self.__setitem__(i, o)

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None:
        ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None:
        ...

    def __delitem__(self, i: int) -> None:
        self.solutions.__delitem__(i)

    def __len__(self) -> int:
        return self.solutions.__len__()

    def damn_sol(self, idx: int = -1):
        self.damned_sols.append(self.solutions.pop(idx))
