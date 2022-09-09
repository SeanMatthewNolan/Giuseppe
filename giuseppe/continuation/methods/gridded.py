from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Iterator, Iterable
from typing import Union, Optional
from copy import copy
from enum import Enum, auto
import queue

import numpy as np

from giuseppe.io import Solution, SolutionSet
from .abstract import ContinuationSeries
from ...utils.exceptions import ContinuationError
from ...utils.typing import NPArray, EMPTY_ARRAY


@dataclass
class CellJob:
    coordinates: tuple[int]
    guess: Solution


class Grid:
    def __init__(self,  min_values: tuple[float], max_values: tuple[float], num_divisions: tuple[int]):
        self.num_divisions: tuple[int] = num_divisions
        self.min_values: tuple[float] = min_values
        self.max_values: tuple[float] = max_values

        self.n_dim = len(self.num_divisions)
        # Verify consistency
        if not (len(self.num_divisions) == len(self.min_values) == len(self.max_values) == self.n_dim):
            raise RuntimeError(
                    'Inconsistency Found: Ensure that there are the same number of elements specified in constant '
                    'names, min values, max values, and number of divisions')

        # Generate the axis of the grid
        self.axes = tuple([
            np.linspace(min_val, max_val, num)
            for min_val, max_val, num in zip(self.min_values, self.max_values, self.num_divisions)
        ])

        self.solutions = np.empty(self.num_divisions, dtype=Solution)

    def get_constants(self, *coordinates: tuple[int]):
        return tuple(axis[coord] for axis, coord in zip(self.axes, coordinates))

    def find_nearest_cell(self, constants: Iterable[float]):
        return tuple(axis[(np.abs(axis - constant)).argmin()] for axis, constant in zip(self.axes, constants))

    def get_solution(self,  *coordinates: tuple[int]):
        return self.solutions[coordinates]

    def set_solution(self, solution: Solution,  *coordinates: tuple[int]):
        self.solutions[coordinates] = solution
        return self.solutions[coordinates]


# TODO Implement multiprocessing version
class Gridded(ContinuationSeries, ABC):
    def __init__(self, constant_names: Iterable[str], min_values: Iterable[float], max_values: Iterable[float],
                 num_divisions: Union[int, Iterable[int]], solution_set: SolutionSet):

        super().__init__(solution_set)

        self.constant_names: Iterable[str] = tuple(constant_names)
        self.constant_indices: tuple[int] = self._get_constant_indices()
        self.n_dim: int = len(self.constant_indices)

        if not isinstance(num_divisions, Iterable):
            num_divisions = [num_divisions] * self.n_dim

        self.num_divisions = tuple(num_divisions)
        self.min_values = tuple(min_values)
        self.max_values = tuple(max_values)

        self.grid = Grid(self.min_values, self.max_values, self.num_divisions)

        # TODO multiprocessing version should use queue.Queue
        self.queue: list = []

        self.base_constants: np.ndarray = EMPTY_ARRAY

    def _get_constant_indices(self) -> tuple[int]:
        indices = []
        for constant_key in self.constant_names:
            try:
                indices.append(self.solution_set.constant_names.index(constant_key))
            except ValueError:
                raise KeyError(f'Cannot perform continuation on {constant_key} because it is not a defined constant')

        return tuple(indices)

    def _get_constants(self, new_values: Iterable[float]):
        new_constants = copy(self.base_constants)
        np.put(new_constants, self.constant_indices, new_values)
        return new_constants

    def __repr__(self):
        return f'GriddedContinuation({self.constant_names})'


class GriddedFill(Gridded):
    def __iter__(self) -> Iterator[tuple[NPArray, Solution]]:
        if len(self.solution_set) == 0:
            raise ContinuationError('No converged solution in solution set! Cannot perform continuation')

        root_solution = self.solution_set[-1]
        self.base_constants = copy(root_solution.k)
        mutable_constants = np.take(self.base_constants, self.constant_indices)
        first_job = CellJob(
                self._get_constants(self.grid.find_nearest_cell(mutable_constants))
        )
        return self

    def __next__(self):
        pass

    def __repr__(self):
        return f'GriddedFill({self.constant_names})'
