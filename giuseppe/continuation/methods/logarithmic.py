from math import sqrt

import numpy as np

from .linear import LinearSeries, BisectionLinearSeries
from ...utils.typing import NPArray


class LogarithmicSeries(LinearSeries):
    def __repr__(self):
        return f'LogarithmicSeries({self.form_mapping_str()})'

    def _initialize_iter(self):
        current_constants = self.solution_set[-1].k

        self._steps = np.array([current_constants] * (self.est_num_steps + 1))

        for idx, constant_target in self._idx_target_pairs:
            self._steps[:, idx] = np.geomspace(current_constants[idx], constant_target, self.est_num_steps + 1)

        self._steps = list(self._steps)


class BisectionLogarithmicSeries(LogarithmicSeries, BisectionLinearSeries):
    __init__ = BisectionLinearSeries.__init__
    _initialize_iter = LogarithmicSeries._initialize_iter
    _perform_iter = BisectionLinearSeries._perform_iter

    def __repr__(self):
        return f'BisectionLogarithmicSeries({self.form_mapping_str()})'

    def _bisect_step(self, last_constants: NPArray, next_constants: NPArray) -> NPArray:
        for idx, _ in self._idx_target_pairs:
            next_constants[idx] = sqrt(next_constants[idx] * last_constants[idx])

        return next_constants
