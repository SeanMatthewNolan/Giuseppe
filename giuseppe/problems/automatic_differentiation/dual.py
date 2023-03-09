from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.data_classes.annotations import Annotations
from giuseppe.problems.protocols import OCP, Adjoints, Dual, DifferentialControlHandler
from .input import ADiffInputProb, ADiffInputInequalityConstraints
from .ocp import ADiffOCP
from .adjoints import ADiffAdjoints
from .utils import ca_wrap, lambdify_ca
from ..symbolic.ocp import SymOCP, StrInputProb


class ADiffDual(ADiffOCP, ADiffAdjoints, Dual):
    def __init__(self, input_data: Union[ADiffInputProb, SymOCP, OCP, StrInputProb]):
        ADiffOCP.__init__(self, input_data)
        self.adjoints = ADiffAdjoints(self)

        self.prob_class = 'dual'

        # Unpack Adjoints class into Dual Class

        self.annotations = self.adjoints.annotations

        self.num_costates = self.adjoints.num_costates
        self.num_initial_adjoints = self.adjoints.num_initial_adjoints
        self.num_terminal_adjoints = self.adjoints.num_terminal_adjoints
        self.num_adjoints = self.adjoints.num_adjoints

        self.ca_costate_dynamics = self.adjoints.ca_costate_dynamics

        self.ca_hamiltonian = self.adjoints.ca_hamiltonian
        self.ca_dh_du = self.adjoints.ca_dh_du
        self.ca_dh_dt = self.adjoints.ca_dh_dt

        self.ca_initial_adjoint_cost = self.adjoints.ca_initial_adjoint_cost
        self.ca_terminal_adjoint_cost = self.adjoints.ca_terminal_adjoint_cost

        self.ca_initial_adjoint_boundary_conditions = self.adjoints.ca_initial_adjoint_boundary_conditions
        self.ca_terminal_adjoint_boundary_conditions = self.adjoints.ca_terminal_adjoint_boundary_conditions

        self.compute_costate_dynamics = lambdify_ca(self.ca_costate_dynamics)

        self.compute_hamiltonian = lambdify_ca(self.ca_hamiltonian)
        self.compute_control_law = lambdify_ca(self.ca_dh_du)

        self.compute_initial_adjoint_boundary_conditions = lambdify_ca(self.ca_initial_adjoint_boundary_conditions)
        self.compute_terminal_adjoint_boundary_conditions = lambdify_ca(self.ca_terminal_adjoint_boundary_conditions)

        self.control_handler = ADiffControlHandler(self)

        # self.dtype = self.dual.dtype
        #
        # self.independent = self.dual.independent
        # self.states = self.dual.states
        # self.costates = self.dual.costates
        # self.controls = self.dual.controls
        # self.parameters = self.dual.parameters
        # self.constants = self.dual.constants
        # self.initial_adjoints = self.dual.initial_adjoints
        # self.terminal_adjoints = self.dual.terminal_adjoints
        #
        # self.num_states = self.dual.num_states
        # self.num_costates = self.dual.num_costates
        # self.num_controls = self.dual.num_controls
        # self.num_parameters = self.dual.num_parameters
        # self.num_constants = self.dual.num_constants
        # self.num_initial_adjoints = self.dual.num_initial_adjoints
        # self.num_terminal_adjoints = self.dual.num_terminal_adjoints
        #
        # self.initial_independent = self.dtype.sym('t_0', 1)
        # self.terminal_independent = self.dtype.sym('t_f', 1)
        # self.tau = self.dtype.sym('τ', 1)
        # _independent = self.tau * (self.terminal_independent - self.initial_independent) + self.initial_independent


        # # Jacobians for the Dynamics
        # self.dependent = ca.vcat((self.states, self.costates, self.controls))
        # self.num_dependent = self.dependent.shape[0]
        #
        # self.bvp_parameters = ca.vcat((
        #     self.parameters,  # p
        #     self.initial_adjoints,  # nu_0
        #     self.terminal_adjoints,  # nu_f
        #     self.initial_independent,  # t_0
        #     self.terminal_independent  # t_f
        # ))
        # self.num_bvp_parameters = self.bvp_parameters.shape[0]
        #
        # _x_dot = self.ocp.ca_dynamics(*self.ocp.dyn_args)
        # _lam_dot = self.dual.ca_costate_dynamics(*self.dual.args['dynamic'])
        # _u_dot = self.control_handler.ca_control_dynamics(*self.dual.args['dynamic'])
        # _y_dot = ca.vcat((_x_dot, _lam_dot, _u_dot))
        # _dt_dtau = self.terminal_independent - self.initial_independent
        # _y_dot_transformed = ca.substitute(_y_dot, self.independent, _independent)
        # _dy_dtau = _y_dot_transformed * _dt_dtau
        #
        # _dyn_y_jac = ca.jacobian(_dy_dtau, self.dependent)
        # _dyn_p_jac = ca.jacobian(_dy_dtau, self.bvp_parameters)
        #
        # _df_dy_out_labs = list()
        # for i in range(self.num_dependent):
        #     _df_dy_out_labs.append(f'df_dy{i+1}')
        #
        # _df_dp_out_labs = list()
        # for i in range(self.num_bvp_parameters):
        #     _df_dp_out_labs.append(f'df_dp{i+1}')
        #
        # self.df_dy = ca.Function('df_dy', (self.tau, self.dependent, self.bvp_parameters, self.constants),
        #                          ca.horzsplit(_dyn_y_jac),
        #                          ('τ', 'y', 'p_nu_t', 'k'), _df_dy_out_labs)
        # self.df_dp = ca.Function('df_dp', (self.tau, self.dependent, self.bvp_parameters, self.constants),
        #                          ca.horzsplit(_dyn_p_jac),
        #                          ('τ', 'y', 'p_nu_t', 'k'), _df_dp_out_labs)
        #
        # # Jacobians for the Boundary Conditions
        # self.initial_dependent = self.dtype.sym('ya', self.num_dependent)
        # self.terminal_dependent = self.dtype.sym('yb', self.num_dependent)
        #
        # self.bc_jac_args = (self.initial_dependent, self.terminal_dependent, self.bvp_parameters, self.constants)
        # self.bc_jac_arg_names = ('ya', 'yb', 'p_nu_t', 'k')
        #
        # _idx_x0 = 0
        # _idx_lam0 = self.num_states
        # _idx_u0 = _idx_lam0 + self.num_costates
        #
        # _x0 = self.initial_dependent[:_idx_lam0]
        # _lam0 = self.initial_dependent[_idx_lam0:_idx_u0]
        # _u0 = self.initial_dependent[_idx_u0:]
        #
        # _xf = self.terminal_dependent[:_idx_lam0]
        # _lamf = self.terminal_dependent[_idx_lam0:_idx_u0]
        # _uf = self.terminal_dependent[_idx_u0:]
        #
        # _bc = ca.vcat((self.ocp.ca_boundary_conditions.initial(self.initial_independent, _x0, _u0,
        #                                                        self.parameters, self.constants),
        #                self.ocp.ca_boundary_conditions.terminal(self.terminal_independent, _xf, _uf,
        #                                                         self.parameters, self.constants),
        #                self.dual.ca_adj_boundary_conditions.initial(self.initial_independent, _x0, _lam0, _u0,
        #                                                             self.parameters, self.initial_adjoints,
        #                                                             self.constants),
        #                self.dual.ca_adj_boundary_conditions.terminal(self.terminal_independent, _xf, _lamf, _uf,
        #                                                              self.parameters, self.terminal_adjoints,
        #                                                              self.constants),
        #                self.control_handler.ca_control_bc(self.initial_independent, _x0, _lam0, _u0,
        #                                                   self.parameters, self.constants)
        #                ))
        #
        # _dbc_dya = ca.jacobian(_bc, self.initial_dependent)
        # _dbc_dyb = ca.jacobian(_bc, self.terminal_dependent)
        # _dbc_dp = ca.jacobian(_bc, self.bvp_parameters)
        #
        # self.dbc_dya = ca.Function('dbc_dya', self.bc_jac_args, (_dbc_dya,), self.bc_jac_arg_names, ('dbc_dya',))
        # self.dbc_dyb = ca.Function('dbc_dyb', self.bc_jac_args, (_dbc_dyb,), self.bc_jac_arg_names, ('dbc_dya',))
        # self.dbc_dp = ca.Function('dbc_dp_nu_t', self.bc_jac_args, (_dbc_dp,), self.bc_jac_arg_names, ('dbc_dp_nu_t',))


class ADiffControlHandler(DifferentialControlHandler):
    def __init__(self, source_dual: ADiffDual):
        self.source_dual = source_dual

        args = self.source_dual.adjoints.args
        arg_names = self.source_dual.adjoints.arg_names

        _h_u = ca.jacobian(self.source_dual.ca_hamiltonian(
                *args['adj_dynamic']), self.source_dual.controls)
        _h_uu = ca.jacobian(_h_u, self.source_dual.controls)
        _h_ut = ca.jacobian(_h_u, self.source_dual.independent)
        _h_ux = ca.jacobian(_h_u, self.source_dual.states)
        _f = self.source_dual.ca_dynamics(*args['ocp_dynamic'])
        _f_u = ca.jacobian(_f, self.source_dual.controls)
        _lam_dot = self.source_dual.ca_costate_dynamics(*args['adj_dynamic'])[:self.source_dual.num_states]

        self.h_u: ca.Function = ca.Function('H_u',  args['adj_dynamic'], (_h_u,), arg_names['adj_dynamic'], ('H_u',))
        self.h_uu: ca.Function = ca.Function('H_uu', args['adj_dynamic'], (_h_uu,), arg_names['adj_dynamic'], ('H_uu',))
        self.h_ut: ca.Function = ca.Function('H_ut', args['adj_dynamic'], (_h_ut,), arg_names['adj_dynamic'], ('H_ut',))
        self.h_ux: ca.Function = ca.Function('H_ux', args['adj_dynamic'], (_h_ux,), arg_names['adj_dynamic'], ('H_ux',))
        self.f_u: ca.Function = ca.Function('f_u', args['ocp_dynamic'], (_f_u,), arg_names['ocp_dynamic'], ('f_u',))

        self.ca_control_dynamics: ca.Function = ca.Function('u_dot', args['adj_dynamic'],
                                                            (-ca.solve(_h_uu, _h_ut + _h_ux @ _f + _f_u.T @ _lam_dot),),
                                                            arg_names['adj_dynamic'], ('u_dot',))
        self.ca_control_bc: ca.Function = ca.Function(
                'u_bc', args['adj_dynamic'], (_h_u.T,), arg_names['adj_dynamic'], ('transpose_H_u',))

        self.compute_control_dynamics = lambdify_ca(self.ca_control_dynamics)
        self.compute_control_boundary_conditions = lambdify_ca(self.ca_control_bc)


# TODO implement AdiffAlgControlHandler using ca.DaeBuilder
# class AdiffAlgControlHandler:
#     def __init__(self, control_handler: CompAlgControlHandler, adiff_dual: AdiffDual):
#         self.adiff_dual = adiff_dual
#         self.control_handler = control_handler
#
#         self.comp_control = self.control_handler.control
#         self.args = self.adiff_dual.args['dynamic']
#         self.iter_args = self.adiff_dual.iter_args['dynamic']
#         self.arg_names = self.adiff_dual.arg_names['dynamic']
#
#         self.ca_control = self.wrap_control()
#
#     def wrap_control(self):
#         return ca_wrap('u', self.args, self.comp_control, self.iter_args, self.arg_names)
