from copy import copy
from typing import Callable, Optional, Union

import numpy as np

from ..typing import NPArray

ScalarFunction = Callable[[float], Union[float, NPArray]]
ArrayFunction = Callable[[NPArray], NPArray]


def forward_difference(func: ScalarFunction, x: float, h: Optional[float] = 1e-6) -> Union[float, NPArray]:
    h *= x
    return (func(x + h) - func(x)) / h


def backward_difference(func: ScalarFunction, x: float, h: Optional[float] = 1e-6) -> Union[float, NPArray]:
    h *= x
    return (func(x) - func(x - h)) / h


def central_difference(func: ScalarFunction, x: float, h: Optional[float] = 1e-6) -> Union[float, NPArray]:
    h *= x
    return (func(x + h/2) - func(x - h/2)) / h


def forward_difference_jacobian(func: ArrayFunction, x: NPArray, h: Optional[float] = 1e-6) -> NPArray:
    f_star = func(x)
    jacobian = np.empty((len(f_star), len(x)))

    for idx, xi in enumerate(x):
        hi = h * xi
        x_step = copy(x)
        x_step[idx] += hi
        jacobian[:, idx] = (func(x_step) - f_star) / hi

    return jacobian


def backward_difference_jacobian(func: ArrayFunction, x: NPArray, h: Optional[float] = 1e-6) -> NPArray:
    return forward_difference_jacobian(func, x, h=-h)


def central_difference_jacobian(func: ArrayFunction, x: NPArray, h: Optional[float] = 1e-6) -> NPArray:
    f_star = func(x)
    jacobian = np.empty((len(f_star), len(x)))

    for idx, xi in enumerate(x):
        hi = h * xi
        x_step_b, x_step_f = copy(x), copy(x)
        x_step_b[idx] -= hi / 2
        x_step_f[idx] += hi / 2
        jacobian[:, idx] = (func(x_step_f) - func(x_step_b)) / hi

    return jacobian
