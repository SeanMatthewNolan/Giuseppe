from __future__ import annotations
from collections.abc import Hashable, Mapping, Iterator, Iterable
from typing import Union, Optional
from copy import copy

import numpy as np

from giuseppe.data_classes import Solution, SolutionSet, Annotations
from .abstract import ContinuationSeries
from ...utils.exceptions import ContinuationError
from ...utils.typing import NPArray


class LinearSeries(ContinuationSeries):
    def __init__(
            self, num_steps: int, target_mapping: Mapping[Hashable: float], solution_set: SolutionSet,
            constant_names: Optional[Union[Iterable[Hashable, ...], Annotations]] = None
    ):

        super().__init__(solution_set)
        self.num_steps: int = num_steps
        self.target_mapping: Mapping[Hashable: float] = target_mapping

        if constant_names is None:
            self.constant_names: tuple[Hashable, ...] = tuple(range(len(self.solution_set[-1].k)))
        elif isinstance(constant_names, Annotations):
            self.constant_names: tuple[Hashable, ...] = tuple(constant_names.constants)
        else:
            self.constant_names: tuple[Hashable, ...] = tuple(constant_names)

        self.constant_indices = self._get_constant_indices()
        self.constant_targets = np.fromiter(self.target_mapping.values(), dtype=float)
        self.idx_tar_pairs: list[tuple[int, float]] = \
            [(idx, tar) for idx, tar in zip(self._get_constant_indices(), target_mapping.values())]

        self._delta: NPArray
        self._step_size: NPArray

    def __iter__(self) -> Iterator[tuple[NPArray, Solution]]:
        if len(self.solution_set) == 0:
            raise ContinuationError('No converged solution in solution set! Cannot perform continuation')

        self.current_step = 0
        self._compute_step_size()

        return self

    def _compute_step_size(self):
        self._delta = self.constant_targets - self.solution_set[-1].k[self.constant_indices]
        self._step_size = self._delta / self.num_steps

    def __next__(self):
        previous_solution = self.solution_set[-1]

        if not previous_solution.converged:
            self.solution_set.damn_sol()
            raise ContinuationError('Previous solution did not converge. Continuation cannot continue.')

        if self.current_step == self.num_steps:
            raise StopIteration

        self.current_step += 1
        next_constants = self._generate_next_constants()

        return next_constants, self.solution_set[-1]

    def _get_constant_indices(self) -> list[int]:
        indices = []
        for constant_key, target_value in self.target_mapping.items():
            try:
                indices.append(self.constant_names.index(constant_key))
            except ValueError:
                raise KeyError(f'Cannot perform continuation on {constant_key} because it is not a defined constant')

        return indices

    def _generate_next_constants(self):
        next_constants = copy(self.solution_set[-1].k)
        next_constants[self.constant_indices] += self._step_size
        return next_constants

    def __repr__(self):
        return f'LinearSeries({self.generate_target_mapping_str()})'

    def generate_target_mapping_str(self):
        return self.generate_mapping_str(self.target_mapping.values())

    def generate_mapping_str(self, values):
        name_str = ', '.join(self.target_mapping.keys())
        val_str = ', '.join(f'{float(val):.2}' for val in values)
        return f'{name_str} -> {val_str}'


class BisectionLinearSeries(LinearSeries):
    def __init__(
            self, num_steps: int, target_mapping: Mapping[Hashable: float], solution_set: SolutionSet,
            max_bisections: int = 3, constant_names: Optional[Union[Iterable[Hashable, ...], Annotations]] = None
    ):

        LinearSeries.__init__(self, num_steps, target_mapping, solution_set, constant_names=constant_names)

        self.max_bisections: int = max_bisections
        self.bisection_counter: int = 0

    def __iter__(self):
        super().__iter__()
        self.bisection_counter = 0
        return self

    def __next__(self):
        if self.solution_set[-1].converged:
            if self.current_step == self.num_steps:
                raise StopIteration

            self.current_step += 1
            next_constants = self._generate_next_constants()

            if self.bisection_counter > 0:
                self.bisection_counter -= 1

        else:
            self.solution_set.damn_sol()
            if self.bisection_counter < self.max_bisections:
                self.bisection_counter += 1
                self.num_steps += 1
                next_constants = self._generate_next_constants()

                print(f'Last continuation {self.generate_mapping_str(self.solution_set[-1].k[self.constant_indices])}'
                      f' did not converge. Bisecting next step (depth = {self.bisection_counter})')

            else:
                raise ContinuationError('Bisection limit exceeded!')

        return next_constants, self.solution_set[-1]

    def _generate_next_constants(self):
        next_constants = copy(self.solution_set[-1].k)
        next_constants[self.constant_indices] += self._step_size * 2 ** -self.bisection_counter
        return next_constants

    def __repr__(self):
        return f'BisectionLinearSeries({self.generate_target_mapping_str()})'
