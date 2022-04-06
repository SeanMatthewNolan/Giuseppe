from typing import Union
from collections.abc import Sized

import numpy as np
from numpy.typing import ArrayLike

from ..problems.bvp import CompBVP, BVPSol
from ..problems.ocp import CompOCP, OCPSol


def generate_ones_ocp_guess(prob: Union[CompBVP, CompOCP], t_span: Union[float, ArrayLike] = 0.1,
                            multiplier: float = 1.) -> Union[BVPSol, OCPSol]:
    if isinstance(t_span, float) or isinstance(t_span, int):
        t = np.array([0., t_span], dtype=float)
    else:
        t = np.array(t_span)

    num_t_steps = len(t)

    x = np.ones((prob.num_states, num_t_steps)) * multiplier
    p = np.ones((prob.num_parameters,)) * multiplier
    k = prob.src_bvp.default_values

    if isinstance(prob, CompBVP):
        return BVPSol(t=t, x=x, p=p, k=k)

    elif isinstance(prob, CompOCP):
        u = np.ones((prob.num_controls, num_t_steps)) * multiplier
        return OCPSol(t=t, x=x, u=u, p=p, k=k)
