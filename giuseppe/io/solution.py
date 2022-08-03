from os.path import splitext
from dataclasses import dataclass
from typing import Optional
import json
import pickle

import bson

from giuseppe.utils.conversion import convert_arrays_to_list_in_dict
from giuseppe.utils.typing import NPArray


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

    aux: Optional[dict] = None

    converged: bool = False

    def as_dict(self, arr_to_list: bool = False):
        sol_dict = {
            't': self.t, 'x': self.x, 'p': self.p, 'k': self.k, 'u': self.u,
            'lam': self.lam, 'nu0': self.nu0, 'nuf': self.nuf, 'aux': self.aux, 'converged': self.converged,
        }

        if arr_to_list:
            sol_dict = convert_arrays_to_list_in_dict(sol_dict)

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
