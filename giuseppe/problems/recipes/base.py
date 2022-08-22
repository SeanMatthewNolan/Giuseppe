from abc import ABC, abstractmethod
from collections.abc import Callable

from ..typing import AnyProblem


class RecipeBase(Callable, ABC):
    def __int__(self):
        pass

    @abstractmethod
    def __call__(self, input_prob: AnyProblem, **kwargs) -> AnyProblem:
        pass
