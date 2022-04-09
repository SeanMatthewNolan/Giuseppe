from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ..typing import AnyProblem
else:
    AnyProblem = TypeVar('AnyProblem')


class Regularizer(ABC):
    @abstractmethod
    def apply(self, prob: AnyProblem, item: Any) -> AnyProblem:
        pass
