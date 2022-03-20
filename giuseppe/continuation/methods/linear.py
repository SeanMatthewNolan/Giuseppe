from copy import copy
from collections.abc import Hashable, Mapping, Iterator

import numpy as np

from .abstract import ContinuationSeries
from ..solution_set import SolutionSet
from giuseppe.problems.bvp import BVPSol
from ...utils.typing import NPArray


class LinearSeries(ContinuationSeries):
    def __init__(self, num_steps: int, target_mapping: Mapping[Hashable: float], solution_set: SolutionSet):

        super().__init__(solution_set)
        self.num_steps: int = num_steps
        self.target_mapping: Mapping[Hashable: float] = target_mapping

        self._steps: list[NPArray]

    def __iter__(self) -> Iterator[tuple[NPArray, BVPSol]]:
        next_guess = self.solution_set.get_last()
        current_constants = next_guess.k
        target_constants = self._generate_target_constants(current_constants)

        self._steps = list(np.linspace(current_constants, target_constants, self.num_steps + 1))

        for current_constants in self._steps[1:]:
            yield current_constants, next_guess
            next_guess = self.solution_set.get_last()

    def _generate_target_constants(self, current_constants: NPArray) -> NPArray:

        target_constants = copy(current_constants)

        for constant_key, target_value in self.target_mapping.items():
            try:
                idx = self.solution_set.constant_names.index(constant_key)
            except ValueError:
                raise KeyError(f'Cannot perform continuation on {constant_key} because it is not a defined constant')

            target_constants[idx] = target_value

        return target_constants
