from typing import Any, Protocol, runtime_checkable, Union

from ..bvp import SymBVP
from ..ocp import SymOCP

Problem = Union[SymBVP, SymOCP]


@runtime_checkable
class SymRegularizer(Protocol):
    def apply(self, prob: Problem, item: Any, position: str) -> Problem:
        pass
