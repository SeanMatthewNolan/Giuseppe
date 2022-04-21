from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.utils.mixins import Picky
from .symbolic import SymOCP
from ..components.adiff import AdiffBoundaryConditions, AdiffCost, ca_wrap
from ..ocp.compiled import CompOCP
from ..ocp.adiff import AdiffOCP


class AdiffDual(Picky):
    SUPPORTED_INPUTS: type = Union[SymOCP, CompOCP, AdiffOCP]  # TODO change to take in SymOCP, CompOCP

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
        elif isinstance(self.src_ocp, SymOCP):
            self.adiff_ocp: AdiffOCP = AdiffOCP(self.src_ocp)

        self.num_states = self.adiff_ocp.num_states
        self.num_controls = self.adiff_ocp.num_controls
        self.num_parameters = self.adiff_ocp.num_parameters
        self.num_constants = self.adiff_ocp.num_constants

        # TODO implement num_costates, num_initial_adjoints, num_terminal_adjoints
        self.num_costates = self.num_states + self.num_parameters
        self.num_initial_adjoints = self.adiff_ocp.ca_boundary_conditions.initial.size_out(0)[0]
        self.num_terminal_adjoints = self.adiff_ocp.ca_boundary_conditions.terminal.size_out(0)[0]

        arg_lens = {'initial': (1, self.num_states, self.num_costates, self.num_controls,
                                self.num_parameters, self.num_initial_adjoints, self.num_constants),
                    'dynamic': (1, self.num_states, self.num_costates, self.num_controls,
                                self.num_parameters, self.num_constants),
                    'terminal': (1, self.num_states, self.num_costates, self.num_controls,
                                 self.num_parameters, self.num_terminal_adjoints, self.num_constants),
                    'ocp': (1, self.num_states, self.num_controls, self.num_parameters, self.num_constants)}

        self.arg_names = {'initial': ('t', 'x', 'lam', 'u', 'p', '_nu_0', 'k'),
                          'dynamic': ('t', 'x', 'lam', 'u', 'p', 'k'),
                          'terminal': ('t', 'x', 'lam', 'u', 'p', '_nu_f', 'k'),
                          'ocp': ('t', 'x', 'u', 'p', 'k')}

        self.args = {}
        self.iter_args = {}
        for key in self.arg_names:
            self.args[key] = [ca.SX.sym(name, length) for name, length in zip(self.arg_names[key], arg_lens[key])]
            self.iter_args[key] = [ca.vertsplit(arg, 1) for arg in self.args[key][1:]]
            self.iter_args[key].insert(0, self.args[key][0])  # Insert time separately b/c not wrapped in list

        self.t = self.args['dynamic'][0]
        self.x = self.args['dynamic'][1]
        self.costates = self.args['dynamic'][2]
        self.lam = self.costates[:self.num_states]
        self.u = self.args['dynamic'][3]
        self.p = self.args['dynamic'][4]
        self.x_and_p = ca.vcat((self.x, self.p))
        self._nu_0 = self.args['initial'][5]
        self._nu_f = self.args['terminal'][5]
        self.k = self.args['dynamic'][5]

        psi0 = self.adiff_ocp.ca_boundary_conditions.initial
        psif = self.adiff_ocp.ca_boundary_conditions.terminal

        self.ca_hamiltonian = ca.Function('H', self.args['dynamic'],
                                          (self.adiff_ocp.ca_cost.path(*self.adiff_ocp.args)
                                           + ca.dot(self.lam, self.adiff_ocp.ca_dynamics(*self.adiff_ocp.args)),),
                                          self.arg_names['dynamic'], ('H',))
        self.ca_costate_dynamics = ca.Function('lam_dot', self.args['dynamic'],
                                               (-ca.jacobian(self.ca_hamiltonian(*self.args['dynamic']),
                                                             self.x_and_p).T,),
                                               self.arg_names['dynamic'], ('lam_dot',))

        initial_aug_cost = ca.Function('Phi_0adj', self.args['initial'],
                                       (self.adiff_ocp.ca_cost.initial(*self.adiff_ocp.args)
                                        + ca.dot(self._nu_0, psi0(*self.adiff_ocp.args)),),
                                       self.arg_names['initial'], ('Phi_0adj',))
        terminal_aug_cost = ca.Function('Phi_fadj', self.args['terminal'],
                                        (self.adiff_ocp.ca_cost.terminal(*self.adiff_ocp.args)
                                         + ca.dot(self._nu_f, psif(*self.adiff_ocp.args)),),
                                        self.arg_names['terminal'], ('Phi_fadj',))
        self.ca_augmented_cost = AdiffCost(initial_aug_cost, self.ca_hamiltonian, terminal_aug_cost)

        # TODO jacobian returning 0 instead of adjoints!
        adj1 = ca.jacobian(initial_aug_cost(*self.args['initial']), self.t) - self.ca_hamiltonian(*self.args['dynamic'])
        adj2 = ca.jacobian(initial_aug_cost(*self.args['initial']), self.x_and_p).T + self.costates
        adj3 = ca.jacobian(terminal_aug_cost(*self.args['terminal']),
                           self.t) + self.ca_hamiltonian(*self.args['dynamic'])
        adj4 = ca.jacobian(terminal_aug_cost(*self.args['terminal']), self.x_and_p).T - self.costates

        initial_adj_bcs = ca.Function('Psi_0adj', self.args['initial'],
                                      (ca.vertcat(adj1, adj2),),
                                      self.arg_names['initial'], ('Psi_0adj',))
        terminal_adj_bcs = ca.Function('Psi_fadj', self.args['terminal'],
                                       (ca.vertcat(adj3, adj4),),
                                       self.arg_names['terminal'], ('Psi_fadj',))
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
        ocp_args = self.adiff_ocp.args
        arg_names = self.adiff_dual.arg_names['dynamic']
        ocp_arg_names = self.adiff_ocp.arg_names

        _h_u = ca.jacobian(self.adiff_dual.ca_hamiltonian(*args), self.adiff_dual.u)
        _h_uu = ca.jacobian(_h_u, self.adiff_dual.u)
        _h_ut = ca.jacobian(_h_u, self.adiff_dual.t)
        _h_ux = ca.jacobian(_h_u, self.adiff_dual.x)
        _f = self.adiff_ocp.ca_dynamics(*ocp_args)
        _f_u = ca.jacobian(_f, self.adiff_dual.u)
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

    # TODO implement ca_control_dynamics
    # def wrap_control_dynamics(self):

    # TODO implement ca_bc
    # def wrap_control_bc(self):


class AdiffDualOCP:
    def __init__(self, adiff_ocp: AdiffOCP, adiff_dual: AdiffDual, control_method: str = 'differential'):
        self.ocp = adiff_ocp
        self.dual = adiff_dual

        if control_method.lower() == 'differential':
            self.control_handler = AdiffDiffControlHandler(self.ocp, self.dual)
        else:
            raise NotImplementedError(
                f'\"{control_method}\" is not an implemented control method. Try \"differential\".')
