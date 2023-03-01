from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.problems.symbolic.mixins import Picky
from giuseppe.problems.dual.symbolic import SymOCP
from giuseppe.problems.input import AdiffInputProb
from giuseppe.problems.automatic_differentiation.regularization import AdiffBoundaryConditions, AdiffCost
from giuseppe.problems.ocp.adiff import AdiffOCP
from giuseppe.problems.ocp.compiled import CompOCP


class AdiffDual(Picky):
    SUPPORTED_INPUTS: type = Union[AdiffInputProb, SymOCP, CompOCP, AdiffOCP]

    def __init__(self, source_ocp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_ocp)

        self.src_ocp = deepcopy(source_ocp)

        if isinstance(self.src_ocp, AdiffOCP):
            self.adiff_ocp: AdiffOCP = deepcopy(self.src_ocp)
        elif isinstance(self.src_ocp, CompOCP):
            if self.src_ocp.use_jit_compile:
                warn('AdiffDual cannot accept JIT compiled CompOCP! Recompiling CompOCP without JIT...')
                self.adiff_ocp: AdiffOCP = AdiffOCP(self.src_ocp.src_ocp)
            else:
                self.adiff_ocp: AdiffOCP = AdiffOCP(self.src_ocp)
        elif isinstance(self.src_ocp, SymOCP) or isinstance(self.src_ocp, AdiffInputProb):
            self.adiff_ocp: AdiffOCP = AdiffOCP(self.src_ocp)
        else:
            raise TypeError(f"AdiffDual cannot be initialized with a {type(source_ocp)} object!")

        self.dtype = self.adiff_ocp.dtype
        self.num_states = self.adiff_ocp.num_states
        self.num_controls = self.adiff_ocp.num_controls
        self.num_parameters = self.adiff_ocp.num_parameters
        self.num_constants = self.adiff_ocp.num_constants
        self.num_costates = self.num_states + self.num_parameters
        self.num_initial_adjoints = self.adiff_ocp.ca_boundary_conditions.initial.size_out(0)[0]
        self.num_terminal_adjoints = self.adiff_ocp.ca_boundary_conditions.terminal.size_out(0)[0]

        self.independent = self.adiff_ocp.independent
        self.states = self.adiff_ocp.states
        self.controls = self.adiff_ocp.controls
        self.parameters = self.adiff_ocp.parameters
        self.constants = self.adiff_ocp.constants

        self.costates = self.dtype.sym('lam', self.num_costates)
        self.initial_adjoints = self.dtype.sym('_nu_0', self.num_initial_adjoints)
        self.terminal_adjoints = self.dtype.sym('_nu_f', self.num_terminal_adjoints)
        self.states_and_parameters = ca.vcat((self.states, self.parameters))

        self.arg_names = {'ocp': self.adiff_ocp.arg_names,
                          'initial': ('t', 'x', 'lam', 'u', 'p', '_nu_0', 'k'),
                          'dynamic': ('t', 'x', 'lam', 'u', 'p', 'k'),
                          'terminal': ('t', 'x', 'lam', 'u', 'p', '_nu_f', 'k')}

        self.args = {'ocp': self.adiff_ocp.args,
                     'initial': (self.independent, self.states, self.costates, self.controls, self.parameters,
                                 self.initial_adjoints, self.constants),
                     'dynamic': (self.independent, self.states, self.costates, self.controls, self.parameters,
                                 self.constants),
                     'terminal': (self.independent, self.states, self.costates, self.controls, self.parameters,
                                  self.terminal_adjoints, self.constants)}

        psi_0 = self.adiff_ocp.ca_boundary_conditions.initial(*self.args['ocp'])
        psi_f = self.adiff_ocp.ca_boundary_conditions.terminal(*self.args['ocp'])
        phi_0 = self.adiff_ocp.ca_cost.initial(*self.args['ocp'])
        phi_f = self.adiff_ocp.ca_cost.terminal(*self.args['ocp'])
        L = self.adiff_ocp.ca_cost.path(*self.args['ocp'])
        f = self.adiff_ocp.eom
        H = L + ca.dot(self.costates, f)
        Hxp = ca.jacobian(H, self.states_and_parameters)
        Hu = ca.jacobian(H, self.controls)
        Ht = ca.jacobian(H, self.independent)
        phi_0_adj = phi_0 + ca.dot(self.initial_adjoints, psi_0)
        phi_f_adj = phi_f + ca.dot(self.terminal_adjoints, psi_f)

        self.ca_hamiltonian = ca.Function('H', self.args['dynamic'], (H,), self.arg_names['dynamic'], ('H',))
        self.ca_costate_dynamics = ca.Function('lam_dot', self.args['dynamic'], (-Hxp.T,),
                                               self.arg_names['dynamic'], ('lam_dot',))
        self.ca_dH_du = ca.Function('dH_du', self.args['dynamic'], (Hu,), self.arg_names['dynamic'], ('dH_du',))
        self.ca_dH_dt = ca.Function('dH_dt', self.args['dynamic'], (Ht,), self.arg_names['dynamic'], ('dH_dt',))

        initial_aug_cost = ca.Function('Phi_0_adj', self.args['initial'], (phi_0_adj,),
                                       self.arg_names['initial'], ('Phi_0_adj',))
        terminal_aug_cost = ca.Function('Phi_f_adj', self.args['terminal'],
                                        (phi_f_adj,),
                                        self.arg_names['terminal'], ('Phi_f_adj',))
        self.ca_augmented_cost = AdiffCost(initial_aug_cost, self.ca_hamiltonian, terminal_aug_cost)

        adj1 = ca.jacobian(phi_0_adj, self.independent) - H
        adj2 = ca.jacobian(phi_0_adj, self.states_and_parameters).T + self.costates
        adj3 = ca.jacobian(phi_f_adj, self.independent) + H
        adj4 = ca.jacobian(phi_f_adj, self.states_and_parameters).T - self.costates

        initial_adj_bcs = ca.Function('Psi_0_adj', self.args['initial'],
                                      (ca.vertcat(adj1, adj2),),
                                      self.arg_names['initial'], ('Psi_0_adj',))
        terminal_adj_bcs = ca.Function('Psi_f_adj', self.args['terminal'],
                                       (ca.vertcat(adj3, adj4),),
                                       self.arg_names['terminal'], ('Psi_f_adj',))
        self.ca_adj_boundary_conditions = AdiffBoundaryConditions(initial_adj_bcs, terminal_adj_bcs)

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


# TODO convert CompDiffControlHandler to AdiffDiffControlHandler -wlevin 4/8/2022
class AdiffDiffControlHandler:
    def __init__(self, adiff_ocp: AdiffOCP, adiff_dual: AdiffDual):
        self.adiff_ocp: AdiffOCP = adiff_ocp
        self.adiff_dual: AdiffDual = adiff_dual

        args = self.adiff_dual.args['dynamic']
        ocp_args = self.adiff_dual.args['ocp']
        arg_names = self.adiff_dual.arg_names['dynamic']
        ocp_arg_names = self.adiff_dual.arg_names['ocp']

        _h_u = ca.jacobian(self.adiff_dual.ca_hamiltonian(*args), self.adiff_dual.controls)
        _h_uu = ca.jacobian(_h_u, self.adiff_dual.controls)
        _h_ut = ca.jacobian(_h_u, self.adiff_dual.independent)
        _h_ux = ca.jacobian(_h_u, self.adiff_dual.states)
        _f = self.adiff_ocp.ca_dynamics(*ocp_args)
        _f_u = ca.jacobian(_f, self.adiff_dual.controls)
        _lam_dot = self.adiff_dual.ca_costate_dynamics(*args)[:self.adiff_ocp.num_states]

        self.h_u: ca.Function = ca.Function('H_u', args, (_h_u,), arg_names, ('H_u',))
        self.h_uu: ca.Function = ca.Function('H_uu', args, (_h_uu,), arg_names, ('H_uu',))
        self.h_ut: ca.Function = ca.Function('H_ut', args, (_h_ut,), arg_names, ('H_ut',))
        self.h_ux: ca.Function = ca.Function('H_ux', args, (_h_ux,), arg_names, ('H_ux',))
        self.f_u: ca.Function = ca.Function('f_u', ocp_args, (_f_u,), ocp_arg_names, ('f_u',))

        self.ca_control_dynamics: ca.Function = ca.Function('u_dot', args,
                                                            (-ca.solve(_h_uu, _h_ut + _h_ux @ _f + _f_u.T @ _lam_dot),),
                                                            arg_names, ('u_dot',))
        self.ca_control_bc: ca.Function = ca.Function('u_bc', args, (_h_u.T,), arg_names, ('transpose_H_u',))


class AdiffDualOCP:
    def __init__(self, adiff_ocp: AdiffOCP, adiff_dual: AdiffDual, control_method: str = 'differential'):
        self.ocp = deepcopy(adiff_ocp)
        self.dual = deepcopy(adiff_dual)

        self.dtype = self.dual.dtype

        self.independent = self.dual.independent
        self.states = self.dual.states
        self.costates = self.dual.costates
        self.controls = self.dual.controls
        self.parameters = self.dual.parameters
        self.constants = self.dual.constants
        self.initial_adjoints = self.dual.initial_adjoints
        self.terminal_adjoints = self.dual.terminal_adjoints

        self.num_states = self.dual.num_states
        self.num_costates = self.dual.num_costates
        self.num_controls = self.dual.num_controls
        self.num_parameters = self.dual.num_parameters
        self.num_constants = self.dual.num_constants
        self.num_initial_adjoints = self.dual.num_initial_adjoints
        self.num_terminal_adjoints = self.dual.num_terminal_adjoints

        self.initial_independent = self.dtype.sym('t_0', 1)
        self.terminal_independent = self.dtype.sym('t_f', 1)
        self.tau = self.dtype.sym('τ', 1)
        _independent = self.tau * (self.terminal_independent - self.initial_independent) + self.initial_independent

        if control_method.lower() == 'differential':
            self.control_handler = AdiffDiffControlHandler(self.ocp, self.dual)
        else:
            raise NotImplementedError(
                f'\"{control_method}\" is not an implemented control method. Try \"differential\".')

        # Jacobians for the Dynamics
        self.dependent = ca.vcat((self.states, self.costates, self.controls))
        self.num_dependent = self.dependent.shape[0]

        self.bvp_parameters = ca.vcat((
            self.parameters,  # p
            self.initial_adjoints,  # nu_0
            self.terminal_adjoints,  # nu_f
            self.initial_independent,  # t_0
            self.terminal_independent  # t_f
        ))
        self.num_bvp_parameters = self.bvp_parameters.shape[0]

        _x_dot = self.ocp.ca_dynamics(*self.ocp.args)
        _lam_dot = self.dual.ca_costate_dynamics(*self.dual.args['dynamic'])
        _u_dot = self.control_handler.ca_control_dynamics(*self.dual.args['dynamic'])
        _y_dot = ca.vcat((_x_dot, _lam_dot, _u_dot))
        _dt_dtau = self.terminal_independent - self.initial_independent
        _y_dot_transformed = ca.substitute(_y_dot, self.independent, _independent)
        _dy_dtau = _y_dot_transformed * _dt_dtau

        _dyn_y_jac = ca.jacobian(_dy_dtau, self.dependent)
        _dyn_p_jac = ca.jacobian(_dy_dtau, self.bvp_parameters)

        _df_dy_out_labs = list()
        for i in range(self.num_dependent):
            _df_dy_out_labs.append(f'df_dy{i+1}')

        _df_dp_out_labs = list()
        for i in range(self.num_bvp_parameters):
            _df_dp_out_labs.append(f'df_dp{i+1}')

        self.df_dy = ca.Function('df_dy', (self.tau, self.dependent, self.bvp_parameters, self.constants),
                                 ca.horzsplit(_dyn_y_jac),
                                 ('τ', 'y', 'p_nu_t', 'k'), _df_dy_out_labs)
        self.df_dp = ca.Function('df_dp', (self.tau, self.dependent, self.bvp_parameters, self.constants),
                                 ca.horzsplit(_dyn_p_jac),
                                 ('τ', 'y', 'p_nu_t', 'k'), _df_dp_out_labs)

        # Jacobians for the Boundary Conditions
        self.initial_dependent = self.dtype.sym('ya', self.num_dependent)
        self.terminal_dependent = self.dtype.sym('yb', self.num_dependent)

        self.bc_jac_args = (self.initial_dependent, self.terminal_dependent, self.bvp_parameters, self.constants)
        self.bc_jac_arg_names = ('ya', 'yb', 'p_nu_t', 'k')

        _idx_x0 = 0
        _idx_lam0 = self.num_states
        _idx_u0 = _idx_lam0 + self.num_costates

        _x0 = self.initial_dependent[:_idx_lam0]
        _lam0 = self.initial_dependent[_idx_lam0:_idx_u0]
        _u0 = self.initial_dependent[_idx_u0:]

        _xf = self.terminal_dependent[:_idx_lam0]
        _lamf = self.terminal_dependent[_idx_lam0:_idx_u0]
        _uf = self.terminal_dependent[_idx_u0:]

        _bc = ca.vcat((self.ocp.ca_boundary_conditions.initial(self.initial_independent, _x0, _u0,
                                                               self.parameters, self.constants),
                       self.ocp.ca_boundary_conditions.terminal(self.terminal_independent, _xf, _uf,
                                                                self.parameters, self.constants),
                       self.dual.ca_adj_boundary_conditions.initial(self.initial_independent, _x0, _lam0, _u0,
                                                                    self.parameters, self.initial_adjoints,
                                                                    self.constants),
                       self.dual.ca_adj_boundary_conditions.terminal(self.terminal_independent, _xf, _lamf, _uf,
                                                                     self.parameters, self.terminal_adjoints,
                                                                     self.constants),
                       self.control_handler.ca_control_bc(self.initial_independent, _x0, _lam0, _u0,
                                                          self.parameters, self.constants)
                       ))

        _dbc_dya = ca.jacobian(_bc, self.initial_dependent)
        _dbc_dyb = ca.jacobian(_bc, self.terminal_dependent)
        _dbc_dp = ca.jacobian(_bc, self.bvp_parameters)

        self.dbc_dya = ca.Function('dbc_dya', self.bc_jac_args, (_dbc_dya,), self.bc_jac_arg_names, ('dbc_dya',))
        self.dbc_dyb = ca.Function('dbc_dyb', self.bc_jac_args, (_dbc_dyb,), self.bc_jac_arg_names, ('dbc_dya',))
        self.dbc_dp = ca.Function('dbc_dp_nu_t', self.bc_jac_args, (_dbc_dp,), self.bc_jac_arg_names, ('dbc_dp_nu_t',))
