from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Union, Optional, Iterable

from giuseppe.data_classes import Solution, SolutionSet, Annotations

from .display import ContinuationDisplayManager, ProgressBarDisplay, NoDisplay
from .methods import ContinuationSeries, LinearSeries, BisectionLinearSeries, LogarithmicSeries, \
    BisectionLogarithmicSeries
from ..numeric_solvers import NumericSolver
from ..utils.exceptions import ContinuationError


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

    def __init__(self, numeric_solver: NumericSolver, root: Union[Solution, SolutionSet],
                 constant_names: Optional[Union[Iterable[Hashable, ...], Annotations]] = None):
        """
        Initialize continuation handler
        """
        self.numeric_solver: NumericSolver = numeric_solver

        if isinstance(root, Solution):
            if not root.converged:
                root = numeric_solver.solve(root.k, root)

                if not root.converged:
                    raise ValueError('Guess for root solution did not converged!!!')

            self.solution_set: SolutionSet = SolutionSet(solutions=[root])

        elif isinstance(root, SolutionSet):
            if len(root) < 1:
                raise ValueError('Please ensure that the root solution set has at least one solution')
            self.solution_set: SolutionSet = root
        else:
            raise TypeError('Please provide continuation handler root solution or guess '
                            'of type Solution or SolutionSet')

        self.continuation_series: list[ContinuationSeries] = []

        if constant_names is None:
            _root_sol = self.solution_set[0]
            if _root_sol.annotations is None:
                self.constant_names: tuple[Hashable, ...] = tuple(range(_root_sol.k))
            else:
                self.constant_names = _root_sol.annotations
        elif isinstance(constant_names, Annotations):
            self.constant_names: tuple[Hashable, ...] = tuple(constant_names.constants)
        else:
            self.constant_names: tuple[Hashable, ...] = tuple(constant_names)

        self.monitor: Optional[ContinuationDisplayManager] = None

    def add_linear_series(self, num_steps: int, target_values: Mapping[Hashable: float],
                          bisection: Union[bool, int] = True):
        """
        Add a linear series to the continuation handler

        The linear series will take linearly spaced steps toward the specified target values using the last solution as
        the next guess

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
            series = BisectionLinearSeries(num_steps, target_values, self.solution_set,
                                           constant_names=self.constant_names)
        elif bisection > 0:
            series = BisectionLinearSeries(num_steps, target_values, self.solution_set,
                                           max_bisections=bisection, constant_names=self.constant_names)
        else:
            series = LinearSeries(num_steps, target_values, self.solution_set, constant_names=self.constant_names)

        self.continuation_series.append(series)
        return self

    def add_logarithmic_series(self, num_steps: int, target_values: Mapping[Hashable: float],
                               bisection: Union[bool, int] = True):
        """
        Add a logarithmic series to the continuation handler

        The logarithmic series will take proportionally spaced steps toward the specified target values using the last
        solution as the next guess

        Parameters
        ----------
        num_steps : int
            number of steps in the continuation series

        target_values : dict[str: float]
           dictionary (or other mapping) assigning target values to continuation series
           key should be the name of the constant to change
           value is the final value

        bisection : Union[bool, int], default=True
           If True or number, the continuation handler will retry to solve problem with bisected step length if solver
           fails to converge.

           If a number is given, the number specifies the maximum number of bisections that will occur before giving up.

        Returns
        -------
        self : ContinuationHandler
        """

        if bisection is True:
            series = BisectionLogarithmicSeries(num_steps, target_values, self.solution_set,
                                                constant_names=self.constant_names)
        elif bisection > 0:
            series = BisectionLogarithmicSeries(num_steps, target_values, self.solution_set, max_bisections=bisection,
                                                constant_names=self.constant_names)
        else:
            series = LogarithmicSeries(num_steps, target_values, self.solution_set,
                                       constant_names=self.constant_names)

        self.continuation_series.append(series)
        return self

    def run_continuation(self, display: Optional[ContinuationDisplayManager] = ProgressBarDisplay()) -> SolutionSet:
        """
        Run continuation set

        Parameters
        ----------
        display : optional

        Returns
        -------
        solution_set

        """

        if display is None:
            display = NoDisplay()

        try:
            with display:
                for series in self.continuation_series:
                    display.start_cont_series(series)
                    for k, last_sol in series:
                        self.solution_set.append(self.numeric_solver.solve(k, last_sol))
                        display.log_step()
                    display.end_cont_series()
            return self.solution_set

        except ContinuationError as e:
            print(f'Continuation failed to complete because exception: {e}')
            return self.solution_set
