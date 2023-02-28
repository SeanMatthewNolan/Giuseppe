from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.utils.typing import SymMatrix
from .input import AdiffInputProb
from ..components.adiff import AdiffBoundaryConditions, ca_wrap


class AdiffBVP:
    def __init__(self, source_bvp: SUPPORTED_INPUTS):
        self.source_bvp = deepcopy(source_bvp)

        self.arg_names = ('t', 'x', 'p', 'k')

        if isinstance(self.source_bvp, AdiffInputProb):
            self.comp_bvp = None

            self.independent = self.source_bvp.independent
            self.states = self.source_bvp.states.states
            self.parameters = self.source_bvp.parameters
            self.constants = self.source_bvp.constants.constants
            self.default_values = self.source_bvp.constants.default_values
            self.eom = self.source_bvp.states.eoms
            self.inputConstraints = self.source_bvp.constraints

            self.num_states = self.states.numel()
            self.num_parameters = self.parameters.numel()
            self.num_constants = self.constants.numel()

            self.args = (self.independent, self.states, self.parameters, self.constants)
            self.ca_dynamics = ca.Function('f', self.args, (self.eom,), self.arg_names, 'dx_dt')
            self.ca_boundary_conditions = self.create_boundary_conditions()
        else:
            if isinstance(self.source_bvp, CompBVP):
                if self.source_bvp.use_jit_compile:
                    warn('AdiffDual cannot accept JIT compiled CompDual! Recompiling CompDual without JIT...')
                    self.comp_bvp: CompBVP = CompBVP(self.source_bvp.src_bvp, use_jit_compile=False)
                else:
                    self.comp_bvp: CompBVP = deepcopy(self.source_bvp)
                self.constants: SymMatrix = self.source_bvp.src_bvp.constants
            elif isinstance(self.source_bvp, SymBVP):
                self.comp_bvp: CompBVP = CompBVP(self.source_bvp, use_jit_compile=False)
                self.constants: SymMatrix = self.source_bvp.constants

            self.num_states = self.comp_bvp.num_states
            self.num_parameters = self.comp_bvp.num_parameters
            self.num_constants = self.comp_bvp.num_constants
            self.default_values = self.comp_bvp.default_values

            self.independent = ca.MX.sym('t', 1)
            self.states = ca.MX.sym('x', self.num_states)
            self.parameters = ca.MX.sym('p', self.num_parameters)
            self.constants = ca.MX.sym('k', self.num_parameters)

            self.args = (self.independent, self.states, self.parameters, self.constants)
            self.iter_args = [ca.vertsplit(arg, 1) for arg in self.args[1:]]
            self.iter_args.insert(0, self.args[0])  # Insert time separately b/c not wrapped in list

            self.ca_dynamics = self.wrap_dynamics()
            self.ca_boundary_conditions = self.wrap_boundary_conditions()

    def wrap_dynamics(self):
        dynamics = ca_wrap('f', self.args, self.comp_bvp.dynamics, self.iter_args,
                           self.arg_names, 'dx_dt')
        return dynamics

    # TODO refactor into AdiffBVP class? -wlevin 4/8/2022
    def wrap_boundary_conditions(self):
        initial_boundary_conditions = ca_wrap('Psi_0', self.args, self.comp_bvp.boundary_conditions.initial,
                                              self.iter_args, self.arg_names)
        terminal_boundary_conditions = ca_wrap('Psi_f', self.args, self.comp_bvp.boundary_conditions.terminal,
                                               self.iter_args, self.arg_names)
        return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    def create_boundary_conditions(self):
        initial_boundary_conditions = ca.Function('Psi_0', self.args, (self.inputConstraints.initial,),
                                                  self.arg_names, ('Psi_0',))
        terminal_boundary_conditions = ca.Function('Psi_f', self.args, (self.inputConstraints.terminal,),
                                                   self.arg_names, ('Psi_f',))
        return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)
