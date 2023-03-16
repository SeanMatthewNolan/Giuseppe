from __future__ import annotations
from collections.abc import Hashable, Mapping, Iterable
from typing import Union, Optional
from copy import copy

import numpy as np

from giuseppe.data_classes import SolutionSet, Annotations
from .abstract import ContinuationSeries
from ...utils.typing import NPArray


class UntilFailureSeries(ContinuationSeries):
    def __init__(
            self, step_mapping: Mapping[Hashable: float], solution_set: SolutionSet,
            max_bisections: int = 3, constant_names: Optional[Union[Iterable[Hashable, ...], Annotations]] = None
    ):

        super().__init__(solution_set)
        self.step_mapping: Mapping[Hashable: float] = step_mapping

        if constant_names is None:
            self.constant_names: tuple[Hashable, ...] = tuple(range(len(self.solution_set[-1].k)))
        elif isinstance(constant_names, Annotations):
            self.constant_names: tuple[Hashable, ...] = tuple(constant_names.constants)
        else:
            self.constant_names: tuple[Hashable, ...] = tuple(constant_names)

        self.constant_indices = self._get_constant_indices()
        self.constant_steps = np.fromiter(self.step_mapping.values(), dtype=float)
        self.idx_tar_pairs: list[tuple[int, float]] = \
            [(idx, tar) for idx, tar in zip(self._get_constant_indices(), step_mapping.values())]

        self._step_size: NPArray
        self.max_bisections: int = max_bisections
        self.bisection_counter: int = 0

    def __iter__(self):
        super().__iter__()
        self.bisection_counter = 0
        return self

    def __next__(self):
        if self.solution_set[-1].converged:
            self.current_step += 1
            next_constants = self._generate_next_constants()

            if self.bisection_counter > 0:
                self.bisection_counter -= 1

        else:
            self.solution_set.damn_sol()
            if self.bisection_counter < self.max_bisections:
                self.bisection_counter += 1
                next_constants = self._generate_next_constants()

            else:
                raise StopIteration

        return next_constants, self.solution_set[-1]

    def __repr__(self):
        return f'UntilFailureSeries({self.generate_target_mapping_str()})'

    def _generate_next_constants(self):
        next_constants = copy(self.solution_set[-1].k)
        next_constants[self.constant_indices] += self.constant_steps * 2 ** -self.bisection_counter
        return next_constants

    def _get_constant_indices(self) -> list[int]:
        indices = []
        for constant_key, target_value in self.step_mapping.items():
            try:
                indices.append(self.constant_names.index(constant_key))
            except ValueError:
                raise KeyError(f'Cannot perform continuation on {constant_key} because it is not a defined constant')

        return indices

    def generate_target_mapping_str(self):
        return self.generate_mapping_str(self.step_mapping.values())

    def generate_mapping_str(self, values):
        name_str = ', '.join(self.step_mapping.keys())
        val_str = ', '.join(f'{float(val):.2}' for val in values)
        return f'{name_str} += {val_str}'
