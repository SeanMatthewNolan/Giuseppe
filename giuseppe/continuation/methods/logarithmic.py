from copy import copy

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
    __next__ = BisectionLinearSeries.__next__
    _compute_step_size = LogarithmicSeries._compute_step_size

    def _generate_next_constants(self):
        next_constants = copy(self.solution_set[-1].k)
        next_constants[self.constant_indices] *= self._step_size ** (2 ** -self.bisection_counter)
        return next_constants

    def __repr__(self):
        return f'BisectionLogarithmicSeries({self.generate_target_mapping_str()})'
