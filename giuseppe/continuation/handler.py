from collections.abc import Hashable, Mapping
from typing import Union

from .methods import ContinuationSeries, LinearSeries, BisectionLinearSeries, LogarithmicSeries, \
    BisectionLogarithmicSeries
from .solution_set import SolutionSet


class ContinuationHandler:
    """
    Class for performing continuation process.

    It is initialized with a solution set in order to have access to all previous solutions for guesses and to output
    results.

    User adds more continuation steps by calling methods on object.

    Attributes
    ----------
    continuation_series : list[ContinuationSeries]
    solution_set : SolutionSet
    constant_names : tuple[Hashable, ...]
    """

    def __init__(self, solution_set: SolutionSet):
        """
        Initialize continuation handler

        Parameters
        ----------
        solution_set: SolutionSet
            Object for continuation handler to retrieve guess and store solutions
        """
        self.continuation_series: list[ContinuationSeries] = []
        self.solution_set: SolutionSet = solution_set
        self.constant_names: tuple[Hashable, ...] = solution_set.constant_names

    def add_linear_series(self, num_steps: int, target_values: Mapping[Hashable: float],
                          bisection: Union[bool, int] = False):
        """
        Add a linear series to the continuation handler

        The linear series will take linearly spaced steps torward the specified target values using the last solution as the next guess

        Parameters
        ----------
        num_steps : int
            number of steps in the continuation series

        target_values : dict[str: float]
           dictionary (or other mapping) assigning target values to continuation series
           key should be the name of the constant to change
           value is the final value

        bisection : Union[bool, int], default=False
           If True or number, the continuation handler will retry to solve problem with bisected step length if solver
           fails to converge.

           If a number is given, the number specifies the maximum number of bisections that will occur before giving up.

        Returns
        -------
        self : ContinuationHandler
        """

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
        """
        Add a logarithmic series to the continuation handler

        The logarithmic series will take proportinally spaced steps torward the specified target values using the last
        solution as the next guess

        Parameters
        ----------
        num_steps : int
            number of steps in the continuation series

        target_values : dict[str: float]
           dictionary (or other mapping) assigning target values to continuation series
           key should be the name of the constant to change
           value is the final value

        bisection : Union[bool, int], default=False
           If True or number, the continuation handler will retry to solve problem with bisected step length if solver
           fails to converge.

           If a number is given, the number specifies the maximum number of bisections that will occur before giving up.

        Returns
        -------
        self : ContinuationHandler
        """

        if bisection is True:
            series = BisectionLogarithmicSeries(num_steps, target_values, self.solution_set)
        elif bisection > 0:
            series = BisectionLogarithmicSeries(num_steps, target_values, self.solution_set, max_bisections=bisection)
        else:
            series = LogarithmicSeries(num_steps, target_values, self.solution_set)

        self.continuation_series.append(series)
        return self
