from collections.abc import Hashable, Mapping
from typing import Union

from .methods import ContinuationSeries, LinearSeries, BisectionLinearSeries, LogarithmicSeries, BisectionLogarithmicSeries
from .solution_set import SolutionSet


class ContinuationHandler:
    def __init__(self, solution_set: SolutionSet):
        self.continuation_series: list[ContinuationSeries] = []
        self.solution_set: SolutionSet = solution_set
        self.constant_names: tuple[Hashable, ...] = solution_set.constant_names

    def add_linear_series(self, num_steps: int, target_values: Mapping[Hashable: float],
                          bisection: Union[bool, int] = False):
        if bisection is True:
            series = BisectionLinearSeries(num_steps, target_values, self.solution_set)
        elif bisection > 0:
            series = BisectionLinearSeries(num_steps, target_values, self.solution_set, max_bisections=bisection)
        else:
            series = LinearSeries(num_steps, target_values, self.solution_set)

        self.continuation_series.append(series)
        return self

    def add_logarithmic_series(self, num_steps: int, target_values: Mapping[Hashable: float],
                               bisection: Union[bool, int] = False):

        if bisection is True:
            series = BisectionLogarithmicSeries(num_steps, target_values, self.solution_set)
        elif bisection > 0:
            series = BisectionLogarithmicSeries(num_steps, target_values, self.solution_set, max_bisections=bisection)
        else:
            series = LogarithmicSeries(num_steps, target_values, self.solution_set)

        self.continuation_series.append(series)
        return self
