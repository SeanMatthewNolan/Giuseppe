from copy import deepcopy
from typing import Optional

import numpy as np

from giuseppe.utils import make_array_slices
from giuseppe.utils.compilation import jit_compile
from giuseppe.utils.typing import NumbaFloat, NumbaArray, NumbaMatrix
from giuseppe.data_classes import Solution, Annotations
from giuseppe.problems.protocols.bvp import BVP
from giuseppe.problems.protocols.dual import Dual
from giuseppe.problems.protocols.control_handlers import AlgebraicControlHandler, DifferentialControlHandler

# TODO Add method to convert OCP to BVP by specifying control law


def convert_dual_to_bvp(dual_prob: Dual) -> 'BVPFromDual':
    return BVPFromDual(dual_prob)


class BVPFromDual(BVP):
    def __init__(self, source_dual: Dual, use_jit_compile: bool = True):
        self.source_dual = deepcopy(source_dual)

        _source_num_states = self.source_dual.num_states
        _source_num_costates = self.source_dual.num_costates
        _source_num_controls = self.source_dual.num_controls

        _source_num_parameters = self.source_dual.num_parameters
        _source_num_initial_adjoints = self.source_dual.num_initial_adjoints
        _source_num_terminal_adjoints = self.source_dual.num_terminal_adjoints
        _source_num_adjoints = self.source_dual.num_adjoints

        self._x_slice, self._lam_slice, self._u_slice = \
            make_array_slices((_source_num_states, _source_num_costates, _source_num_controls))
        self._p_slice, self._nu0_slice, self._nuf_slice = \
            make_array_slices((_source_num_parameters, _source_num_initial_adjoints, _source_num_terminal_adjoints))
        self._nu_slice = slice(self._nu0_slice.start, self._nuf_slice.stop)

        self.num_states: int = _source_num_states + _source_num_costates
        self.num_parameters: int = _source_num_parameters + _source_num_adjoints
        self.num_constants: int = self.source_dual.num_constants

        self.default_values: np.ndarray = self.source_dual.default_values
        self.dual_annotations: Optional[Annotations] = self.source_dual.annotations

        if self.dual_annotations is None:
            self.annotations = None
        else:
            self.annotations = Annotations(
                    independent=self.dual_annotations.independent,
                    states=(*self.dual_annotations.states, *self.dual_annotations.costates),
                    parameters=self.dual_annotations.parameters,
                    constants=self.dual_annotations.constants,
                    expressions=self.dual_annotations.expressions
            )

        if isinstance(self.source_dual.control_handler, AlgebraicControlHandler):

            self.compute_dynamics = self._alg_convert_dynamics()
            self.compute_boundary_conditions = self._alg_convert_boundary_conditions()

            self.preprocess_data = self._alg_compile_preprocess()
            self.post_process_data = self._alg_compile_post_process()

        elif isinstance(self.source_dual.control_handler, DifferentialControlHandler):
            if self.dual_annotations is not None:
                self.annotations.states = (*self.annotations.states, *self.dual_annotations.controls)

            self.num_states += _source_num_controls

            self.compute_dynamics = self._diff_convert_dynamics()
            self.compute_boundary_conditions = self._diff_convert_boundary_conditions()

            self.preprocess_data = self._diff_compile_preprocess()
            self.post_process_data = self._diff_compile_post_process()

        else:
            raise ValueError('Cannot convert Dual to BVP without valid control handler')

        if use_jit_compile:
            self.compute_dynamics = jit_compile(
                    self.compute_dynamics, (NumbaFloat, NumbaArray, NumbaArray, NumbaArray)
            )
            self.compute_boundary_conditions = jit_compile(
                    self.compute_boundary_conditions, (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray)
            )

    def _alg_convert_dynamics(self):
        _x_slice, _lam_slice, _p_slice = self._x_slice, self._lam_slice, self._p_slice

        _compute_state_dynamics = self.source_dual.compute_dynamics
        _compute_costate_dynamics = self.source_dual.compute_costate_dynamics
        _compute_control = self.source_dual.control_handler.compute_control

        def compute_dynamics(
                t: float, y: np.ndarray, rho: np.ndarray, k: np.ndarray
        ) -> np.ndarray:

            x = y[_x_slice]
            lam = y[_lam_slice]

            p = rho[_p_slice]

            u = _compute_control(t, x, lam, p, k)

            _x_dot = _compute_state_dynamics(t, x, u, p, k)
            _lam_dot = _compute_costate_dynamics(t, x, lam, u, p, k)

            return np.concatenate((_x_dot, _lam_dot))

        return compute_dynamics

    def _alg_convert_boundary_conditions(self):
        _x_slice, _lam_slice, _u_slice, _p_slice, _nu_slice = \
            self._x_slice, self._lam_slice, self._u_slice, self._p_slice, self._nu_slice

        _compute_state_boundary_conditions = self.source_dual.compute_boundary_conditions
        _compute_adjoint_boundary_conditions = self.source_dual.compute_adjoint_boundary_conditions
        _compute_control = self.source_dual.control_handler.compute_control

        def compute_boundary_conditions(
                t: np.ndarray, y: np.ndarray, rho: np.ndarray, k: np.ndarray
        ) -> np.ndarray:

            x = y[_x_slice, :]
            lam = y[_lam_slice, :]

            p = rho[_p_slice]
            nu = rho[_nu_slice]

            u = np.vstack((
                _compute_control(t[0], x[:, 0], lam[:, 0], p, k),
                _compute_control(t[0], x[:, 0], lam[:, 0], p, k)
            )).T

            _psi = _compute_state_boundary_conditions(t, x, p, k)
            _adj_bc = _compute_adjoint_boundary_conditions(t, x, lam, u, p, nu, k)

            return np.concatenate((_psi, _adj_bc))

        return compute_boundary_conditions

    def _alg_compile_preprocess(self):
        _annotations = self.annotations

        def preprocess_data(in_data: Solution) -> Solution:
            t = in_data.t
            x = np.vstack((in_data.x, in_data.lam))
            p = np.concatenate((in_data.p, in_data.nu0, in_data.nuf))
            k = in_data.k

            lam, u, nu0, nuf = None, None, None, None

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu0, nuf=nuf, k=k,
                            converged=in_data.converged, annotations=_annotations)

        return preprocess_data

    def _alg_compile_post_process(self):
        _x_slice, _lam_slice, _u_slice, _p_slice, _nu0_slice, _nuf_slice = \
            self._x_slice, self._lam_slice, self._u_slice, self._p_slice, self._nu0_slice, self._nuf_slice

        _annotations = self.dual_annotations

        _compute_control = self.source_dual.control_handler.compute_control

        def post_process_data(in_data: Solution) -> Solution:
            t = in_data.t
            x = in_data.x[_x_slice, :]
            lam = in_data.x[_lam_slice, :]
            p = in_data.p[_p_slice]
            nu0 = in_data.p[_nu0_slice]
            nuf = in_data.p[_nuf_slice]
            k = in_data.k

            u = np.array([_compute_control(ti, xi, lam_i, p, k) for ti, xi, lam_i in zip(t, x.T, lam.T)]).T

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu0, nuf=nuf, k=k,
                            converged=in_data.converged, annotations=_annotations)

        return post_process_data

    def _diff_convert_dynamics(self):
        _x_slice, _lam_slice, _u_slice, _p_slice, _nu_slice = \
            self._x_slice, self._lam_slice, self._u_slice, self._p_slice, self._nu_slice

        _compute_state_dynamics = self.source_dual.compute_dynamics
        _compute_costate_dynamics = self.source_dual.compute_costate_dynamics
        _compute_control_dynamics = self.source_dual.control_handler.compute_control_dynamics

        def compute_dynamics(
                t: float, y: np.ndarray, rho: np.ndarray, k: np.ndarray
        ) -> np.ndarray:

            x = y[_x_slice]
            lam = y[_lam_slice]
            u = y[_u_slice]

            p = rho[_p_slice]

            _x_dot = _compute_state_dynamics(t, x, u, p, k)
            _lam_dot = _compute_costate_dynamics(t, x, lam, u, p, k)
            _u_dot = _compute_control_dynamics(t, x, lam, u, p, k)

            return np.concatenate((_x_dot, _lam_dot, _u_dot))

        return compute_dynamics

    def _diff_convert_boundary_conditions(self):
        _x_slice, _lam_slice, _u_slice, _p_slice, _nu_slice = \
            self._x_slice, self._lam_slice, self._u_slice, self._p_slice, self._nu_slice

        _compute_state_boundary_conditions = self.source_dual.compute_boundary_conditions
        _compute_adjoint_boundary_conditions = self.source_dual.compute_adjoint_boundary_conditions
        _compute_control_boundary_conditions = self.source_dual.control_handler.compute_control_dynamics

        def compute_boundary_conditions(
                t: np.ndarray, y: np.ndarray, rho: np.ndarray, k: np.ndarray
        ) -> np.ndarray:

            x = y[_x_slice, :]
            lam = y[_lam_slice, :]
            u = y[_u_slice, :]

            p = rho[_p_slice]
            nu = rho[_nu_slice]

            _psi = _compute_state_boundary_conditions(t, x, p, k)
            _adj_bc = _compute_adjoint_boundary_conditions(t, x, lam, u, p, nu, k)
            _dh_du = _compute_control_boundary_conditions(t[0], x[:, 0], lam[:, 0], u[:, 0], p, k)

            return np.concatenate((_psi, _adj_bc, _dh_du))

        return compute_boundary_conditions

    def _diff_compile_preprocess(self):
        _annotations = self.annotations

        def preprocess_data(in_data: Solution) -> Solution:
            t = in_data.t
            x = np.vstack((in_data.x, in_data.lam, in_data.u))
            p = np.concatenate((in_data.p, in_data.nu0, in_data.nuf))
            k = in_data.k

            lam, u, nu0, nuf = None, None, None, None

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu0, nuf=nuf, k=k,
                            converged=in_data.converged, annotations=_annotations)

        return preprocess_data

    def _diff_compile_post_process(self):
        _x_slice, _lam_slice, _u_slice, _p_slice, _nu0_slice, _nuf_slice = \
            self._x_slice, self._lam_slice, self._u_slice, self._p_slice, self._nu0_slice, self._nuf_slice

        _annotations = self.dual_annotations

        def post_process_data(in_data: Solution) -> Solution:
            t = in_data.t
            x = in_data.x[_x_slice, :]
            lam = in_data.x[_lam_slice, :]
            u = in_data.x[_u_slice, :]
            p = in_data.p[_p_slice]
            nu0 = in_data.p[_nu0_slice]
            nuf = in_data.p[_nuf_slice]
            k = in_data.k

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu0, nuf=nuf, k=k,
                            converged=in_data.converged, annotations=_annotations)

        return post_process_data
