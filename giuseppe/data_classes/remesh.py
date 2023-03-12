import copy

import numpy as np
from scipy.interpolate import PchipInterpolator

from giuseppe.utils.slicing import make_array_slices

from . solution import Solution


# TODO Consider more interpolators
# TODO Make work for non-dual problems
def remesh(in_sol: Solution, t_values: np.ndarray) -> Solution:
    if in_sol.lam is None or in_sol.u is None:
        raise NotImplementedError(
                'This only works for Dual solutions. I (Sean) was wasting too much time trying to make this work for'
                ' BVP and OCP solutions in a clever way. Feel free to add the support if you see this.')

    sol = copy.deepcopy(in_sol)

    dyna_vecs = [in_sol.x, in_sol.lam, in_sol.u]
    x_slice, lam_slice, u_slice = make_array_slices(tuple(_vec.shape[0] for _vec in dyna_vecs))

    dyna_vec = np.vstack(dyna_vecs)

    interpolator = PchipInterpolator(sol.t, dyna_vec, axis=1, extrapolate=True)

    inter_vec = interpolator(t_values)

    sol.t = t_values
    sol.x = inter_vec[x_slice, :]
    sol.lam = inter_vec[lam_slice, :]
    sol.u = inter_vec[u_slice, :]

    return sol
