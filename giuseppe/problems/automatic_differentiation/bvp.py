from copy import deepcopy
from typing import Union, Optional
from warnings import warn

import casadi as ca
from numba.core.registry import CPUDispatcher

from giuseppe.data_classes.annotations import Annotations
from giuseppe.problems.protocols import BVP
from .input import ADiffInputProb
from giuseppe.problems.automatic_differentiation.components import ADiffBoundaryConditions, ca_wrap
from ..symbolic.bvp import SymBVP


class ADiffBVP(BVP):
    def __init__(self, source_bvp: Union[ADiffInputProb, BVP]):
        self.source_bvp = deepcopy(source_bvp)

        self.arg_names = ('t', 'x', 'p', 'k')

        if isinstance(self.source_bvp, ADiffInputProb):
            self.independent = self.source_bvp.independent
            self.states = self.source_bvp.states.states
            self.parameters = self.source_bvp.parameters
            self.constants = self.source_bvp.constants.constants
            self.default_values = self.source_bvp.constants.default_values
            self.eom = self.source_bvp.states.eoms
            self.input_constraints = self.source_bvp.constraints

            self.num_states = self.states.numel()
            self.num_parameters = self.parameters.numel()
            self.num_constants = self.constants.numel()

            self.args = (self.independent, self.states, self.parameters, self.constants)
            self.compute_dynamics = ca.Function('f', self.args, (self.eom,), self.arg_names, 'dx_dt')
            self.compute_boundary_conditions = self.create_boundary_conditions()

        elif isinstance(self.source_bvp, (BVP, SymBVP)):
            if isinstance(self.source_bvp, SymBVP):
                self.source_bvp = self.source_bvp.compile(use_jit_compile=False)

            if isinstance(self.source_bvp.compute_dynamics, CPUDispatcher) \
                    or isinstance(self.source_bvp.compute_boundary_conditions, CPUDispatcher):
                warn('ADiffBVP cannot accept JIT compiled BVP! Please don\'t JIT compile in this case')

            if isinstance(self.source_bvp.compute_dynamics, CPUDispatcher):
                self.source_bvp.compute_dynamics = self.source_bvp.compute_dynamics.py_func

            if isinstance(self.source_bvp.compute_boundary_conditions, CPUDispatcher):
                self.source_bvp.compute_boundary_conditions = self.source_bvp.compute_boundary_conditions.py_func

            self.constants: Optional[Annotations] = self.source_bvp.annotations

            self.num_states = self.source_bvp.num_states
            self.num_parameters = self.source_bvp.num_parameters
            self.num_constants = self.source_bvp.num_constants
            self.default_values = self.source_bvp.default_values

            self.independent = ca.MX.sym('t', 1)
            self.states = ca.MX.sym('x', self.num_states)
            self.parameters = ca.MX.sym('p', self.num_parameters)
            self.constants = ca.MX.sym('k', self.num_constants)

            self.args = (self.independent, self.states, self.parameters, self.constants)
            self.iter_args = [ca.vertsplit(arg, 1) for arg in self.args[1:]]
            self.iter_args.insert(0, self.args[0])  # Insert time separately b/c not wrapped in list

            self.compute_dynamics = self.wrap_dynamics()
            self.compute_boundary_conditions = self.wrap_boundary_conditions()

        else:
            raise ValueError('Need a source BVP')

    def wrap_dynamics(self):
        return ca_wrap('f', self.args, self.source_bvp.compute_dynamics, self.iter_args, self.arg_names, 'dx_dt')

    def wrap_boundary_conditions(self):
        return ca_wrap('Psi', self.iter_args, self.source_bvp.compute_boundary_conditions, self.iter_args,
                       self.arg_names)

    def create_boundary_conditions(self):
        initial_boundary_conditions = ca.Function('Psi_0', self.args, (self.input_constraints.initial,),
                                                  self.arg_names, ('Psi_0',))
        terminal_boundary_conditions = ca.Function('Psi_f', self.args, (self.input_constraints.terminal,),
                                                   self.arg_names, ('Psi_f',))
        return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    # def wrap_boundary_conditions(self):
    #     initial_boundary_conditions = ca_wrap('Psi_0', self.args, self.source_bvp.boundary_conditions.initial,
    #                                           self.iter_args, self.arg_names)
    #     terminal_boundary_conditions = ca_wrap('Psi_f', self.args, self.source_bvp.boundary_conditions.terminal,
    #                                            self.iter_args, self.arg_names)
    #     return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)
    #
    # def create_boundary_conditions(self):
    #     initial_boundary_conditions = ca.Function('Psi_0', self.args, (self.inputConstraints.initial,),
    #                                               self.arg_names, ('Psi_0',))
    #     terminal_boundary_conditions = ca.Function('Psi_f', self.args, (self.inputConstraints.terminal,),
    #                                                self.arg_names, ('Psi_f',))
    #     return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)
