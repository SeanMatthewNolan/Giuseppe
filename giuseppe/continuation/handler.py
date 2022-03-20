from collections.abc import Hashable, Mapping

from .methods import ContinuationSeries, LinearSeries
from .solution_set import SolutionSet


class ContinuationHandler:
    def __init__(self, solution_set: SolutionSet):
        self.continuation_series: list[ContinuationSeries] = []
        self.solution_set: SolutionSet = solution_set
        self.constant_names: tuple[Hashable, ...] = solution_set.constant_names

    def add_linear_series(self, num_steps: int, target_values: Mapping[Hashable: float]):
        self.continuation_series.append(LinearSeries(num_steps, target_values, self.solution_set))
        return self
