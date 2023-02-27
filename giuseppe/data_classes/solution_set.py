# from __future__ import annotations as _annotations
import json
import pprint
import pickle
from abc import abstractmethod
from collections.abc import Iterable, MutableSequence
from os.path import splitext
from typing import Optional, overload, Sequence

import bson

from .solution import Solution
from .annotations import Annotations


class SolutionSet(MutableSequence):
    def __init__(
            self,
            solutions: Optional[Iterable[Solution]] = None,
            annotations: Optional[Annotations] = None,
            check_consistency: bool = True,
    ):

        self.solutions: list[Solution] = []
        self.continuation_slices: list[slice] = []
        self.damned_sols: list[Solution] = []  # Pun: solutions which are not converged

        self.annotations: Optional[Annotations] = annotations
        self.check_consistency = check_consistency

        if solutions is not None:
            if isinstance(solutions, Solution):
                solutions = [solutions]

            if self.annotations is None:
                self.annotations = solutions[0].annotations

            for solution in solutions:
                if solution.converged:
                    self.solutions.append(solution)
                else:
                    self.damned_sols.append(solution)

    def insert(self, index: int, solution: Solution) -> None:
        if self.check_consistency:
            self.perform_consistency_check(solution)
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

    def perform_consistency_check(self, solution):
        if solution.annotations != self.annotations:
            raise ValueError('Latest solution does not consistent with solution set')

    def as_dict(self, arr_to_list: bool = False):
        # TODO Add some more attributes which would be useful with plotting/analysis
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
def load(filename: str = 'sol_set.json', file_format: Optional[str] = None) -> SolutionSet:
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
