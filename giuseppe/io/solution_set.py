import json
import pprint
import pickle
import warnings
from abc import abstractmethod
from collections.abc import Iterable, MutableSequence, Hashable
from copy import deepcopy
from os.path import splitext
from typing import Optional, Union, overload

import bson

from giuseppe.problems.bvp import SymBVP, AdiffBVP
from giuseppe.problems.dual import SymDualOCP, AdiffDual, AdiffDualOCP
from giuseppe.problems.ocp import SymOCP, AdiffOCP
from giuseppe.utils.mixins import Picky
from .solution import Solution


# TODO: add annotations to solution set
class SolutionSet(MutableSequence, Picky):
    SUPPORTED_INPUTS = Union[SymBVP, SymOCP, SymDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP]

    def __init__(self, problem: SUPPORTED_INPUTS, seed_solution: Solution):
        Picky.__init__(self, problem)

        problem = deepcopy(problem)
        if type(problem) is SymDualOCP:
            self.constants = problem.ocp.constants
        elif isinstance(problem, AdiffDualOCP):
            self.constants = problem.dual.adiff_ocp.constants
        elif isinstance(problem, AdiffDual):
            self.constants = problem.adiff_ocp.constants
        elif isinstance(problem, AdiffOCP):
            self.constants = problem.constants
        else:
            self.constants = problem.constants

        if not seed_solution.converged:
            warnings.warn(
                'Seed solution is not converged! It is suggested to solve seed prior to initialization of solution set.'
            )

        self.seed_solution: Solution = seed_solution

        self.solutions: list[Solution] = [seed_solution]
        self.continuation_slices: list[slice] = []
        self.damned_sols: list[Solution] = []

        # Annotations
        self.constant_names: tuple[Hashable, ...] = tuple(str(constant) for constant in self.constants)

    def __repr__(self):
        return f'SolutionSet<{len(self)} Solutions>'

    def insert(self, index: int, solution: Solution) -> None:
        self.solutions.insert(index, solution)

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> Solution:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[Solution]:
        ...

    def __getitem__(self, i: int) -> Solution:
        return self.solutions.__getitem__(i)

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: Solution) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[Solution]) -> None:
        ...

    def __setitem__(self, i: int, o: Solution) -> None:
        self.__setitem__(i, o)

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None:
        ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None:
        ...

    def __delitem__(self, i: int) -> None:
        self.solutions.__delitem__(i)

    def __len__(self) -> int:
        return self.solutions.__len__()

    def damn_sol(self, idx: int = -1):
        self.damned_sols.append(self.solutions.pop(idx))

    def as_dict(self, arr_to_list: bool = False):
        # TODO Add some more attributes which would be useful with plotting/ananlysis
        sol_set_dict = {
            'solutions': [sol.as_dict(arr_to_list=arr_to_list) for sol in self.solutions],
            'damned_sols': [sol.as_dict(arr_to_list=arr_to_list) for sol in self.damned_sols],
        }
        return sol_set_dict

    def as_list_of_dicts(self, arr_to_list: bool = False):
        return [sol.as_dict(arr_to_list=arr_to_list) for sol in self]

    def save(self, filename: str = 'sol_set.json', file_format: Optional[str] = None):
        if file_format is None:
            file_ext = splitext(filename)[1].lower()
            if file_ext == '.json':
                file_format = 'json'
            elif file_ext in ['.bin', '.data', '.pickle']:
                file_format = 'pickle'
            elif file_ext in ['.dict']:
                file_format = 'pickle_dict'
            elif file_ext == '.bson':
                file_format = 'bson'
            elif file_ext == '.zip':
                file_format = 'zip_json'
            elif file_ext in ['.tar', '.gz', '.bz2', '.xz']:
                file_format = 'tar_json'
            else:
                # Here is the default if user specifies neither format nor gives an extension.
                # Also, if the given extension doesn't match
                file_format = 'json'
                filename += '.json'

        file_format = file_format.lower()
        if file_format == 'json':
            self._save_json(filename)
        elif file_format == 'bson':
            self._save_bson(filename)
        elif file_format == 'pickle':
            self._save_pickle(filename)
        elif file_format == 'pickle_dict':
            self._save_pickle_dict(filename)
        elif file_format == 'pickle_no_deps':
            self._save_pickle_no_deps(filename)
        elif file_format.startswith('zip_'):
            self._save_zip(filename)
        elif file_format.startswith('tar_'):
            self._save_tar(filename)
        else:
            raise RuntimeError(f'File format \'{file_format}\' is not an option')

    def _save_json(self, filename: str):
        # with open(filename, 'w') as file:
        #     json.dump(self.as_list_of_dicts(arr_to_list=True), file)

        # Convert python format to JSON
        # (double quotes for strings, lower case for booleans, conversion of tuples to lists, and null for None)
        pp = pprint.PrettyPrinter(indent=4, width=120, compact=True)
        file_text = pp.pformat(self.as_list_of_dicts(arr_to_list=True))
        file_text = file_text.replace('\'', '\"') \
            .replace('False', 'false').replace('True', 'true') \
            .replace('(', '[').replace(')', ']') \
            .replace('None', 'null')
        with open(filename, 'w') as file:
            file.write(file_text)

    def _save_bson(self, filename: str):
        with open(filename, 'wb') as file:
            file.write(bson.dumps(self.as_dict(arr_to_list=True)))

    def _save_pickle(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def _save_pickle_dict(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self.as_list_of_dicts(), file)

    def _save_pickle_no_deps(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self.as_list_of_dicts(arr_to_list=True), file)

    def _save_zip(self, filename: str):
        raise NotImplementedError

    def _save_tar(self, filename: str):
        raise NotImplementedError


def _load_json(filename: str):
    with open(filename, 'r') as file:
        return json.load(file)


def _load_bson(filename: str):
    with open(filename, 'rb') as file:
        return bson.loads(file.read())


def _load_pickle(filename: str):
    with open(filename, 'rb') as file:
        return pickle.load(file)


# TODO Consider embedding metadata into files to distinguish loading files of ambiguous type
def load(filename: str = 'sol.json', file_format: Optional[str] = None) -> Solution:
    if file_format is None:
        file_ext = splitext(filename)[1].lower()
        if file_ext == '.json':
            file_format = 'json'
        elif file_ext == '.bson':
            file_format = 'bson'
        elif file_ext in ['.bin', '.data', '.pickle', '.dict']:
            file_format = 'pickle'
        else:
            raise RuntimeError(
                f'Cannot determine file format automatically: Please specify \'file_format\' manually')

    file_format = file_format.lower()
    if file_format == 'json':
        return _load_json(filename)
    elif file_format == 'bson':
        return _load_bson(filename)
    elif file_format == 'pickle':
        return _load_pickle(filename)
    else:
        raise RuntimeError(f'File format \'{file_format}\' is not an option')
