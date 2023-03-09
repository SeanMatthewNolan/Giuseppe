from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.data_classes.annotations import Annotations
from giuseppe.problems.protocols import OCP, Adjoints
from .input import ADiffInputProb, ADiffInputInequalityConstraints
from .ocp import ADiffOCP
from .utils import ca_wrap, lambdify_ca
from ..symbolic.ocp import SymOCP, StrInputProb


class ADiffAdjoints(Adjoints):
    def __init__(self, source_ocp: Union[ADiffOCP, ADiffInputProb, SymOCP, OCP, StrInputProb]):

        if isinstance(source_ocp, ADiffOCP):
            self.source_ocp = deepcopy(source_ocp)
        else:
            self.source_ocp = ADiffOCP(source_ocp)

        self.annotations = self.source_ocp.annotations

        self.dtype = self.source_ocp.dtype
        self.num_states = self.source_ocp.num_states
        self.num_controls = self.source_ocp.num_controls
        self.num_parameters = self.source_ocp.num_parameters
        self.num_constants = self.source_ocp.num_constants
        self.num_costates = self.num_states + self.num_parameters
        self.num_initial_adjoints = self.source_ocp.ca_initial_boundary_conditions.size_out(0)[0]
        self.num_terminal_adjoints = self.source_ocp.ca_terminal_boundary_conditions.size_out(0)[0]

        self.independent = self.source_ocp.independent
        self.states = self.source_ocp.states
        self.controls = self.source_ocp.controls
        self.parameters = self.source_ocp.parameters
        self.constants = self.source_ocp.constants

        self.costates = self.dtype.sym('lam', self.num_costates)
        self.initial_adjoints = self.dtype.sym('_nu_0', self.num_initial_adjoints)
        self.terminal_adjoints = self.dtype.sym('_nu_f', self.num_terminal_adjoints)
        self.states_and_parameters = ca.vcat((self.states, self.parameters))

        self.arg_names = {
            'ocp_dynamic' : self.source_ocp.dyn_arg_names,
            'ocp_bc'      : self.source_ocp.bc_arg_names,
            'adj_initial' : ('t', 'x', 'lam', 'u', 'p', '_nu_0', 'k'),
            'adj_dynamic' : ('t', 'x', 'lam', 'u', 'p', 'k'),
            'adj_terminal': ('t', 'x', 'lam', 'u', 'p', '_nu_f', 'k')
        }

        self.args = {
            'ocp_dynamic' : self.source_ocp.dyn_args,
            'ocp_bc'      : self.source_ocp.bc_args,
            'adj_initial' : (self.independent, self.states, self.costates, self.controls, self.parameters,
                             self.initial_adjoints, self.constants),
            'adj_dynamic' : (self.independent, self.states, self.costates, self.controls, self.parameters, self.constants),
            'adj_terminal': (self.independent, self.states, self.costates, self.controls, self.parameters,
                             self.terminal_adjoints, self.constants)
        }

        psi_0 = self.source_ocp.ca_initial_boundary_conditions(*self.args['ocp_bc'])
        psi_f = self.source_ocp.ca_terminal_boundary_conditions(*self.args['ocp_bc'])
        phi_0 = self.source_ocp.ca_terminal_cost(*self.args['ocp_bc'])
        phi_f = self.source_ocp.ca_initial_cost(*self.args['ocp_bc'])
        lagrangian = self.source_ocp.ca_path_cost(*self.args['ocp_dynamic'])
        f = self.source_ocp.eom
        hamiltonian = lagrangian + ca.dot(self.costates, f)
        dh_dxp = ca.jacobian(hamiltonian, self.states_and_parameters)
        dh_du = ca.jacobian(hamiltonian, self.controls)
        dh_dt = ca.jacobian(hamiltonian, self.independent)
        phi_0_adj = phi_0 + ca.dot(self.initial_adjoints, psi_0)
        phi_f_adj = phi_f + ca.dot(self.terminal_adjoints, psi_f)

        self.ca_hamiltonian = ca.Function(
                'H', self.args['adj_dynamic'],
                (hamiltonian,),
                self.arg_names['adj_dynamic'], ('H',))
        self.ca_costate_dynamics = ca.Function(
                'lam_dot', self.args['adj_dynamic'],
                (-dh_dxp.T,),
                self.arg_names['adj_dynamic'], ('lam_dot',)
        )
        self.ca_dh_du = ca.Function(
                'dH_du', self.args['adj_dynamic'],
                (dh_du,),
                self.arg_names['adj_dynamic'], ('dH_du',))
        self.ca_dh_dt = ca.Function(
                'dH_dt', self.args['adj_dynamic'],
                (dh_dt,),
                self.arg_names['adj_dynamic'], ('dH_dt',))

        self.ca_initial_adjoint_cost = ca.Function(
                'Phi_0_adj', self.args['adj_initial'],
                (phi_0_adj,),
                self.arg_names['adj_initial'], ('Phi_0_adj',))
        self.ca_terminal_adjoint_cost = ca.Function(
                'Phi_f_adj', self.args['adj_terminal'],
                (phi_f_adj,),
                self.arg_names['adj_terminal'], ('Phi_f_adj',))

        adj1 = ca.jacobian(phi_0_adj, self.independent) - hamiltonian
        adj2 = ca.jacobian(phi_0_adj, self.states_and_parameters).T + self.costates
        adj3 = ca.jacobian(phi_f_adj, self.independent) + hamiltonian
        adj4 = ca.jacobian(phi_f_adj, self.states_and_parameters).T - self.costates

        self.ca_initial_adjoint_boundary_conditions = ca.Function(
                'Psi_0_adj', self.args['adj_initial'],
                (ca.vertcat(adj1, adj2),),
                self.arg_names['adj_initial'], ('Psi_0_adj',)
        )
        self.terminal_adjoint_boundary_conditions = ca.Function(
                'Psi_f_adj', self.args['adj_terminal'],
                (ca.vertcat(adj3, adj4),),
                self.arg_names['adj_terminal'], ('Psi_f_adj',)
        )

        self.compute_hamiltonian = lambdify_ca(self.ca_hamiltonian)
        self.compute_costate_dynamics = lambdify_ca(self.ca_costate_dynamics)

