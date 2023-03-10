import json
import pickle
from dataclasses import dataclass
from os.path import splitext
from typing import Optional

import bson

from giuseppe.utils.conversion import arrays_to_lists_in_dict, lists_to_arrays_in_dict
from giuseppe.utils.typing import NPArray

from .annotations import Annotations


@dataclass
class Solution:
    t: Optional[NPArray] = None
    x: Optional[NPArray] = None
    p: Optional[NPArray] = None
    k: Optional[NPArray] = None

    u: Optional[NPArray] = None

    lam: Optional[NPArray] = None
    nu0: Optional[NPArray] = None
    nuf: Optional[NPArray] = None

    cost: Optional[float] = None

    aux: Optional[dict] = None

    converged: bool = False

    annotations: Optional[Annotations] = None

    def as_dict(self, arr_to_list: bool = False):
        sol_dict = {
            't': self.t, 'x': self.x, 'p': self.p, 'k': self.k, 'u': self.u,
            'lam': self.lam, 'nu0': self.nu0, 'nuf': self.nuf, 'aux': self.aux, 'converged': self.converged,
        }

        if arr_to_list:
            sol_dict = arrays_to_lists_in_dict(sol_dict)

        return sol_dict

    def save(self, filename: str = 'sol.json', file_format: Optional[str] = None):
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
        else:
            raise RuntimeError(f'File format \'{file_format}\' is not an option')

    def _save_json(self, filename: str):
        with open(filename, 'w') as file:
            json.dump(self.as_dict(arr_to_list=True), file)

    def _save_bson(self, filename: str):
        with open(filename, 'wb') as file:
            file.write(bson.dumps(self.as_dict(arr_to_list=True)))

    def _save_pickle(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def _save_pickle_dict(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self.as_dict(), file)

    def _save_pickle_no_deps(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self.as_dict(arr_to_list=True), file)

    # TODO: Look into using Protocol Buffers


def _load_json(filename: str):
    with open(filename, 'r') as file:
        return Solution(**lists_to_arrays_in_dict(json.load(file)))


def _load_bson(filename: str):
    with open(filename, 'rb') as file:
        return Solution(**lists_to_arrays_in_dict(bson.loads(file.read())))


def _load_pickle(filename: str):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def _load_pickle_dict(filename: str):
    with open(filename, 'rb') as file:
        return Solution(**lists_to_arrays_in_dict(pickle.load(file)))


def load(filename: str = 'sol.json', file_format: Optional[str] = None) -> Solution:
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
        else:
            raise RuntimeError(f'Cannot determine file format automatically: Please specify \'file_format\' manually')

    file_format = file_format.lower()
    if file_format == 'json':
        return _load_json(filename)
    elif file_format == 'bson':
        return _load_bson(filename)
    elif file_format == 'pickle':
        return _load_pickle(filename)
    elif file_format in ['pickle_dict', 'pickle_no_deps']:
        return _load_pickle_dict(filename)
    else:
        raise RuntimeError(f'File format \'{file_format}\' is not an option')
