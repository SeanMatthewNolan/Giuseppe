from __future__ import annotations

import sys
from typing import TypeVar, Callable

import numpy as np

from giuseppe.data_classes import Solution


_scipy_bvp_sol = TypeVar('_scipy_bvp_sol')
if sys.version_info >= (3, 10):
    _dyn_type = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    _bc_type = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    _preprocess_type = Callable[[Solution], tuple[np.ndarray, np.ndarray, np.ndarray]]
    _post_process_type = Callable[[_scipy_bvp_sol, np.ndarray], Solution]
else:
    _dyn_type = Callable
    _bc_type = Callable
    _preprocess_type = Callable
    _post_process_type = Callable
