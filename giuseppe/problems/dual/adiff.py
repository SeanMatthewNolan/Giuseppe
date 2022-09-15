from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.utils.mixins import Picky
from .symbolic import SymOCP
from ..components.adiff import AdiffBoundaryConditions, AdiffCost
from ..ocp.adiff import AdiffOCP, AdiffInputOCP
from ..ocp.compiled import CompOCP


class AdiffDual(Picky):
    SUPPORTED_INPUTS: type = Union[AdiffInputOCP, SymOCP, CompOCP, AdiffOCP]

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
        elif isinstance(self.src_ocp, SymOCP) or isinstance(self.src_ocp, AdiffInputOCP):
            self.adiff_ocp: AdiffOCP = AdiffOCP(self.src_ocp)
        else:
            raise TypeError(f"AdiffDual cannot be initialized with a {type(source_ocp)} object!")

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

        self.costates = ca.MX.sym('lam', self.num_costates)
        self.initial_adjoints = ca.MX.sym('_nu_0', self.num_initial_adjoints)
        self.terminal_adjoints = ca.MX.sym('_nu_f', self.num_terminal_adjoints)
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
        phi_0_adj = phi_0 + ca.dot(self.initial_adjoints, psi_0)
        phi_f_adj = phi_f + ca.dot(self.terminal_adjoints, psi_f)

        self.ca_hamiltonian = ca.Function('H', self.args['dynamic'], (H,), self.arg_names['dynamic'], ('H',))
        self.ca_costate_dynamics = ca.Function('lam_dot', self.args['dynamic'], (-Hxp.T,),
                                               self.arg_names['dynamic'], ('lam_dot',))

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
                                                            (-ca.inv(_h_uu)
                                                             @ (_h_ut + _h_ux @ _f + _f_u.T @ _lam_dot),),
                                                            arg_names, ('u_dot',))
        self.ca_control_bc = self.h_u


class AdiffDualOCP:
    def __init__(self, adiff_ocp: AdiffOCP, adiff_dual: AdiffDual, control_method: str = 'differential'):
        self.ocp = deepcopy(adiff_ocp)
        self.dual = deepcopy(adiff_dual)

        self.independent = self.dual.independent
        self.states = self.dual.states
        self.costates = self.dual.costates
        self.controls = self.dual.controls
        self.parameters = self.dual.parameters
        self.constants = self.dual.constants
        self.initial_adjoints = self.dual.initial_adjoints
        self.terminal_adjoints = self.dual.terminal_adjoints

        self.initial_independent = ca.MX.sym('t_0', 1)
        self.terminal_independent = ca.MX.sym('t_f', 1)
        self.tau = ca.MX.sym('τ', 1)
        _independent = self.tau * (self.terminal_independent - self.initial_independent) + self.initial_independent

        if control_method.lower() == 'differential':
            self.control_handler = AdiffDiffControlHandler(self.ocp, self.dual)
        else:
            raise NotImplementedError(
                f'\"{control_method}\" is not an implemented control method. Try \"differential\".')

        self.dyn_jac_args = self.dual.args['dynamic']
        self.dyn_jac_arg_names = self.dual.arg_names['dynamic']
        self.param_jac_args = (self.tau, self.states, self.costates, self.controls, self.parameters,
                               self.initial_adjoints, self.terminal_adjoints,
                               self.initial_independent, self.terminal_independent, self.constants)
        self.param_jac_arg_names = ('τ', 'x', 'lam', 'u', 'p', '_nu_0', '_nu_f', 't_0', 't_f', 'k')

        _x_dot = self.ocp.ca_dynamics(*self.ocp.args)
        _lam_dot = self.dual.ca_costate_dynamics(*self.dual.args['dynamic'])
        _u_dot = self.control_handler.ca_control_dynamics(*self.dual.args['dynamic'])
        _y_dot = ca.vcat((_x_dot, _lam_dot, _u_dot))

        _dyn_y_jac = ca.jacobian(_y_dot, ca.vcat((self.states,  # x
                                                  self.costates,  # lam
                                                  self.controls)))  # u

        # # Derivative w.r.t p, nu_0, nu_f, t_0, t_f (NOTE: t_0, t_f req. subs with t)
        # _dyn_p_jac = ca.vcat(
        #     (ca.jacobian(_y_dot, ca.vcat((self.parameters,  # p
        #                                   self.initial_adjoints,  # nu_0
        #                                   self.terminal_adjoints))),  # nu_f
        #      ca.jacobian(ca.substitute(_y_dot, self.independent, self.initial_independent), self.initial_independent),
        #      ca.jacobian(ca.substitute(_y_dot, self.independent, self.terminal_independent), self.terminal_independent))
        # )

        _dyn_p_jac = ca.jacobian(ca.substitute(_y_dot, self.independent, _independent),
                                 ca.vcat((
                                     self.parameters,  # p
                                     self.initial_adjoints,  # nu_0
                                     self.terminal_adjoints,  # nu_f
                                     self.initial_independent,  # t_0
                                     self.terminal_independent  # t_f
                                 ))
                                 )

        # TODO add bc_jac to AdiffDualOCP

        self.dyn_y_jac = ca.Function('dy_dx_lam_u', self.dyn_jac_args, (_dyn_y_jac,),
                                     self.dyn_jac_arg_names, ('dy_dx_lam_u',))
        self.dyn_p_jac = ca.Function('dy_dp_nu0_nuf', self.param_jac_args, (_dyn_p_jac,),
                                     self.param_jac_arg_names, ('dy_dp_nu0_nuf',))
