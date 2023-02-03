from typing import Any, Protocol, runtime_checkable, TypeVar

Problem = TypeVar('Problem')


@runtime_checkable
class Regularizer(Protocol):
    def apply(self, prob: Problem, item: Any, position: str) -> Problem:
        pass
