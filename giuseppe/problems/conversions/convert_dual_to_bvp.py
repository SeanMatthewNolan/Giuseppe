from copy import deepcopy

import numpy as np

from giuseppe.utils.compilation import jit_compile
from giuseppe.utils.typing import NumbaFloat, NumbaArray, NumbaMatrix
from giuseppe.data_classes import Solution
from giuseppe.problems.protocols.bvp import BVP
from giuseppe.problems.protocols.dual import Dual
from giuseppe.problems.protocols.control_handlers import AlgebraicControlHandler, DifferentialControlHandler

# TODO Add method to convert OCP to BVP by specifying control law


def convert_dual_to_bvp(dual_prob: Dual) -> 'BVPFromDual':
    return BVPFromDual(dual_prob)


class BVPFromDual(BVP):
    def __init__(self, source_dual: Dual, use_jit_compile: bool = True):
        self.source_dual = deepcopy(source_dual)

        if isinstance(self.source_dual.control_handler, AlgebraicControlHandler):
            self.num_states: int = self.source_dual.num_states + self.source_dual.num_costates

            self.compute_dynamics = self._alg_convert_dynamics()
            self.compute_boundary_conditions = self._alg_convert_boundary_conditions()

            self.preprocess_data = self._alg_compile_preprocess()
            self.post_process_data = self._alg_compile_post_process()

        elif isinstance(self.source_dual.control_handler, DifferentialControlHandler):
            self.num_states: int = self.source_dual.num_states + self.source_dual.num_costates \
                                   + self.source_dual.num_controls

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

        self.num_parameters: int = self.source_dual.num_parameters + self.source_dual.num_adjoints
        self.num_constants: int = self.source_dual.num_constants

        self.default_values: np.ndarray = self.source_dual.default_values

    def _alg_convert_dynamics(self):

        _num_states = self.source_dual.num_states
        _num_costates = self.source_dual.num_costates
        _num_parameters = self.source_dual.num_parameters

        _compute_state_dynamics = self.source_dual.compute_dynamics
        _compute_costate_dynamics = self.source_dual.compute_costate_dynamics
        _compute_control = self.source_dual.control_handler.compute_control

        def compute_dynamics(
                t: float, y: np.ndarray, rho: np.ndarray, k: np.ndarray
        ) -> np.ndarray:

            x = y[:_num_states]
            lam = y[_num_states:_num_states + _num_costates]

            p = rho[:_num_parameters]

            u = _compute_control(t, x, lam, p, k)

            _x_dot = _compute_state_dynamics(t, x, u, p, k)
            _lam_dot = _compute_costate_dynamics(t, x, lam, u, p, k)

            return np.concatenate((_x_dot, _lam_dot))

        return compute_dynamics

    def _alg_convert_boundary_conditions(self):
        _num_states = self.source_dual.num_states
        _num_costates = self.source_dual.num_costates

        _num_parameters = self.source_dual.num_parameters

        _compute_state_boundary_conditions = self.source_dual.compute_boundary_conditions
        _compute_adjoint_boundary_conditions = self.source_dual.compute_adjoint_boundary_conditions
        _compute_control = self.source_dual.control_handler.compute_control

        def compute_boundary_conditions(
                t: np.ndarray, y: np.ndarray, rho: np.ndarray, k: np.ndarray
        ) -> np.ndarray:

            x = y[:_num_states, :]
            lam = y[_num_states:_num_states + _num_costates, :]

            p = rho[:_num_parameters]
            nu = rho[_num_parameters:]

            u = np.asarray([_compute_control(ti, xi, lam_i, p, k) for ti, xi, lam_i in zip(t, x.T, lam.T)])

            _psi = _compute_state_boundary_conditions(t, x, p, k)
            _adj_bc = _compute_adjoint_boundary_conditions(t, x, lam, u, p, nu, k)

            return np.concatenate((_psi, _adj_bc))

        return compute_boundary_conditions

    def _alg_compile_preprocess(self):
        _num_states = self.source_dual.num_states
        _num_costates = self.source_dual.num_costates

        _num_parameters = self.source_dual.num_parameters
        _num_initial_adjoints = self.source_dual.num_initial_adjoints
        _num_terminal_adjoints = self.source_dual.num_terminal_adjoints

        _compute_control = self.source_dual.control_handler.compute_control

        def preprocess_data(in_data: Solution) -> Solution:
            t = in_data.t
            x = np.vstack((in_data.x, in_data.lam))
            p = np.concatenate((in_data.p, in_data.nu0, in_data.nuf))
            k = in_data.k

            lam, u, nu0, nuf = None, None, None, None

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu0, nuf=nuf, k=k)

        return preprocess_data

    def _alg_compile_post_process(self):
        _num_states = self.source_dual.num_states
        _num_costates = self.source_dual.num_costates

        _num_parameters = self.source_dual.num_parameters
        _num_initial_adjoints = self.source_dual.num_initial_adjoints
        _num_terminal_adjoints = self.source_dual.num_terminal_adjoints
        _num_adjoints = self.source_dual.num_adjoints

        _compute_control = self.source_dual.control_handler.compute_control

        def post_process_data(in_data: Solution) -> Solution:
            t = in_data.t
            x = in_data.x[:_num_states]
            lam = in_data.x[_num_states:_num_states + _num_costates]
            p = in_data.p[:_num_parameters]
            nu0 = in_data.p[_num_parameters:_num_parameters + _num_initial_adjoints]
            nuf = in_data.p[_num_parameters + _num_initial_adjoints: _num_parameters + _num_adjoints]
            k = in_data.k

            u = np.array(_compute_control(ti, xi, lam_i, p, k) for ti, xi, lam_i in zip(t, x.T, lam.T))

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu0, nuf=nuf, k=k)

        return post_process_data

    def _diff_convert_dynamics(self):

        _num_states = self.source_dual.num_states
        _num_costates = self.source_dual.num_costates

        _num_parameters = self.source_dual.num_parameters

        _compute_state_dynamics = self.source_dual.compute_dynamics
        _compute_costate_dynamics = self.source_dual.compute_costate_dynamics
        _compute_control_dynamics = self.source_dual.control_handler.compute_control_dynamics

        def compute_dynamics(
                t: float, y: np.ndarray, rho: np.ndarray, k: np.ndarray
        ) -> np.ndarray:

            x = y[:_num_states]
            lam = y[_num_states:_num_states + _num_costates]
            u = y[_num_states + _num_costates:]

            p = rho[:_num_parameters]

            _x_dot = _compute_state_dynamics(t, x, u, p, k)
            _lam_dot = _compute_costate_dynamics(t, x, lam, u, p, k)
            _u_dot = _compute_control_dynamics(t, x, lam, u, p, k)

            return np.concatenate((_x_dot, _lam_dot, _u_dot))

        return compute_dynamics

    def _diff_convert_boundary_conditions(self):

        _num_states = self.source_dual.num_states
        _num_costates = self.source_dual.num_costates

        _num_parameters = self.source_dual.num_parameters

        _compute_state_boundary_conditions = self.source_dual.compute_boundary_conditions
        _compute_adjoint_boundary_conditions = self.source_dual.compute_adjoint_boundary_conditions
        _compute_control_boundary_conditions = self.source_dual.control_handler.compute_control_dynamics

        def compute_boundary_conditions(
                t: np.ndarray, y: np.ndarray, rho: np.ndarray, k: np.ndarray
        ) -> np.ndarray:

            x = y[:_num_states, :]
            lam = y[_num_states:_num_states + _num_costates, :]
            u = y[_num_states + _num_costates:, :]

            p = rho[:_num_parameters]
            nu = rho[_num_parameters:]

            _psi = _compute_state_boundary_conditions(t, x, p, k)
            _adj_bc = _compute_adjoint_boundary_conditions(t, x, lam, u, p, nu, k)
            _dh_du = _compute_control_boundary_conditions(t[0], x[:, 0], lam[:, 0], u[:, 0], p, k)

            return np.concatenate((_psi, _adj_bc, _dh_du))

        return compute_boundary_conditions

    def _diff_compile_preprocess(self):
        _num_states = self.source_dual.num_states
        _num_costates = self.source_dual.num_costates

        _num_parameters = self.source_dual.num_parameters
        _num_initial_adjoints = self.source_dual.num_initial_adjoints
        _num_terminal_adjoints = self.source_dual.num_terminal_adjoints

        def preprocess_data(in_data: Solution) -> Solution:
            t = in_data.t
            x = np.vstack((in_data.x, in_data.lam, in_data.u))
            p = np.concatenate((in_data.p, in_data.nu0, in_data.nuf))
            k = in_data.k

            lam, u, nu0, nuf = None, None, None, None

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu0, nuf=nuf, k=k)

        return preprocess_data

    def _diff_compile_post_process(self):
        _num_states = self.source_dual.num_states
        _num_costates = self.source_dual.num_costates
        _num_controls = self.source_dual.num_controls

        _num_parameters = self.source_dual.num_parameters
        _num_initial_adjoints = self.source_dual.num_initial_adjoints
        _num_terminal_adjoints = self.source_dual.num_terminal_adjoints
        _num_adjoints = self.source_dual.num_adjoints

        def post_process_data(in_data: Solution) -> Solution:
            t = in_data.t
            x = in_data.x[:_num_states]
            lam = in_data.x[_num_states:_num_states + _num_costates]
            u = in_data.x[_num_states + _num_costates:_num_states + _num_costates + _num_controls]
            p = in_data.p[:_num_parameters]
            nu0 = in_data.p[_num_parameters:_num_parameters + _num_initial_adjoints]
            nuf = in_data.p[_num_parameters + _num_initial_adjoints:_num_parameters + _num_adjoints]
            k = in_data.k

            return Solution(t=t, x=x, lam=lam, u=u, p=p, nu0=nu0, nuf=nuf, k=k)

        return post_process_data
