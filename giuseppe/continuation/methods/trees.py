from collections.abc import Iterator, Iterable
from typing import Union

import numpy as np

from giuseppe.io import Solution, SolutionSet
from .abstract import ContinuationSeries
from ...utils.typing import NPArray


class RRT(ContinuationSeries):
    def __init__(self, constant_names: Iterable[str], min_values: Iterable[float], max_values: Iterable[float],
                 num_divisions: Union[int, Iterable[int]], solution_set: SolutionSet, max_iterations: int = 1_000):

        super().__init__(solution_set)

        self.constant_names: Iterable[str] = tuple(constant_names)
        self.constant_indices = self._get_constant_indices()
        self.n_dim: int = len(self.constant_indices)

        if not isinstance(num_divisions, Iterable):
            num_divisions = [num_divisions] * self.n_dim

        self.num_divisions = tuple(num_divisions)
        self.min_values = tuple(min_values)
        self.max_values = tuple(max_values)

        # Verify consistency
        if not (self.num_divisions == self.min_values == self.max_values == self.n_dim):
            raise RuntimeError(
                    'Inconsistency Found: Ensure that there are the same number of elements specified in constant '
                    'names, min values, max values, and number of divisions')

        # Generate the axis of the grid
        self.axes = tuple([
            np.linspace(min_val, max_val, num)
            for min_val, max_val, num in zip(self.min_values, self.max_values, self.num_divisions)
        ])

    def _get_constant_indices(self) -> list[int]:
        indices = []
        for constant_key in self.constant_names:
            try:
                indices.append(self.solution_set.constant_names.index(constant_key))
            except ValueError:
                raise KeyError(f'Cannot perform continuation on {constant_key} because it is not a defined constant')

        return indices

    def __iter__(self) -> Iterator[tuple[NPArray, Solution]]:
        pass

    def __next__(self) -> tuple[NPArray, Solution]:
        pass

    def __repr__(self):
        return f'GraphSearch({self.constant_names})'
