from copy import copy

import numpy as np

from .linear import LinearSeries, BisectionLinearSeries
from ...utils.exceptions import ContinuationError


class LogarithmicSeries(LinearSeries):
    def __repr__(self):
        return f'LogarithmicSeries({self.generate_target_mapping_str()})'

    def _compute_step_size(self):
        current_constants = self.solution_set[-1].k[self.constant_indices]
        if any(current_constants <= 0):
            raise ContinuationError('Starting constants to change must be positive for logaritmic continuation')
        self._delta = self.constant_targets / self.solution_set[-1].k[self.constant_indices]
        self._step_size = self._delta ** (1 / self.num_steps)

    def _generate_next_constants(self):
        next_constants = copy(self.solution_set[-1].k)
        next_constants[self.constant_indices] *= self._step_size
        return next_constants


class BisectionLogarithmicSeries(LogarithmicSeries, BisectionLinearSeries):
    __init__ = BisectionLinearSeries.__init__
    __iter__ = BisectionLinearSeries.__iter__
    # __next__ = BisectionLinearSeries.__next__
    _generate_next_constants = BisectionLinearSeries._generate_next_constants
    _compute_step_size = LogarithmicSeries._compute_step_size

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
                self.steps = np.insert(
                        self.steps, self.current_step,
                        (self.steps[self.current_step - 1, :] * self.steps[self.current_step, :]) ** 0.5, axis=0
                )
                next_constants = self._generate_next_constants()

                # print(f'Last continuation {self.generate_mapping_str(self.solution_set[-1].k[self.constant_indices])}'
                #       f' did not converge. Bisecting next step (depth = {self.bisection_counter})')

            else:
                raise ContinuationError('Bisection limit exceeded!')

        return next_constants, self.solution_set[-1]

    def __repr__(self):
        return f'BisectionLogarithmicSeries({self.generate_target_mapping_str()})'
