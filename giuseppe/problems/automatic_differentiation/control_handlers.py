from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

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

