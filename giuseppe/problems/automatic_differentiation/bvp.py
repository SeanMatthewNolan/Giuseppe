from copy import deepcopy
from typing import Union, Optional
from warnings import warn

import casadi as ca
from numba.core.registry import CPUDispatcher

from giuseppe.data_classes.annotations import Annotations
from giuseppe.problems.protocols import BVP
from .input import ADiffInputProb
from .utils import ca_wrap, lambdify_ca
from ..symbolic.bvp import SymBVP, StrInputProb


class ADiffBVP(BVP):
    def __init__(self, source_bvp: Union[ADiffInputProb, SymBVP, BVP]):
        self.source_bvp = deepcopy(source_bvp)

        self.arg_names = ('t', 'x', 'p', 'k')

        if isinstance(self.source_bvp, ADiffInputProb):
            self.dtype = self.source_bvp.dtype

            self.independent = self.source_bvp.independent
            self.states = self.source_bvp.states.states
            self.parameters = self.source_bvp.parameters
            self.constants = self.source_bvp.constants.constants
            self.default_values = self.source_bvp.constants.default_values
            self.eom = self.source_bvp.states.eoms
            self.input_constraints = self.source_bvp.constraints

            self.annotations: Annotations = self.source_bvp.create_annotations()

            self.num_states = self.states.numel()
            self.num_parameters = self.parameters.numel()
            self.num_constants = self.constants.numel()

            self.args = (self.independent, self.states, self.parameters, self.constants)
            self.ca_dynamics = ca.Function('f', self.args, (self.eom,), self.arg_names, ('dx_dt',))
            self.ca_initial_boundary_conditions, self.ca_terminal_boundary_conditions \
                = self.create_boundary_conditions()

        elif isinstance(self.source_bvp, (BVP, SymBVP, StrInputProb)):
            self.dtype = ca.SX

            if isinstance(self.source_bvp, StrInputProb):
                self.source_bvp = SymBVP(self.source_bvp)

            if isinstance(self.source_bvp, SymBVP):
                self.source_bvp = self.source_bvp.compile(use_jit_compile=False)

            if isinstance(self.source_bvp.compute_dynamics, CPUDispatcher) \
                    or isinstance(self.source_bvp.compute_boundary_conditions, CPUDispatcher):
                warn('ADiffBVP cannot accept JIT compiled BVP! Please don\'t JIT compile in this case')

            self.annotations = self.source_bvp.annotations

            if isinstance(self.source_bvp.compute_dynamics, CPUDispatcher):
                self.source_bvp.compute_dynamics = self.source_bvp.compute_dynamics.py_func

            if isinstance(self.source_bvp.compute_boundary_conditions, CPUDispatcher):
                self.source_bvp.compute_boundary_conditions = self.source_bvp.compute_boundary_conditions.py_func

            self.num_states = self.source_bvp.num_states
            self.num_parameters = self.source_bvp.num_parameters
            self.num_constants = self.source_bvp.num_constants
            self.default_values = self.source_bvp.default_values

            self.independent = self.dtype.sym(self.annotations.independent, 1)
            self.states = self.dtype.sym(str(self.annotations.states), self.num_states)
            self.parameters = self.dtype.sym(str(self.annotations.parameters), self.num_parameters)
            self.constants = self.dtype.sym(str(self.annotations.constants), self.num_constants)

            self.args = (self.independent, self.states, self.parameters, self.constants)
            self.iter_args = [ca.vertsplit(arg, 1) for arg in self.args[1:]]
            self.iter_args.insert(0, self.args[0])  # Insert time separately b/c not wrapped in list

            self.ca_dynamics = self.wrap_dynamics()
            self.ca_initial_boundary_conditions, self.ca_terminal_boundary_conditions \
                = self.wrap_boundary_conditions()

        else:
            raise ValueError('Need a source BVP')

        self.compute_dynamics = lambdify_ca(self.ca_dynamics)
        self.compute_initial_boundary_conditions = lambdify_ca(self.ca_initial_boundary_conditions)
        self.compute_terminal_boundary_conditions = lambdify_ca(self.ca_terminal_boundary_conditions)

    def wrap_dynamics(self):
        return ca_wrap('f', self.args, self.source_bvp.compute_dynamics, self.iter_args, self.arg_names, 'dx_dt')

    def create_boundary_conditions(self):
        initial_boundary_conditions = ca.Function('Psi_0', self.args, (self.input_constraints.initial,),
                                                  self.arg_names, ('Psi_0',))
        terminal_boundary_conditions = ca.Function('Psi_f', self.args, (self.input_constraints.terminal,),
                                                   self.arg_names, ('Psi_f',))
        return initial_boundary_conditions, terminal_boundary_conditions

    def wrap_boundary_conditions(self):
        initial_boundary_conditions = ca_wrap('Psi_0', self.args, self.source_bvp.compute_initial_boundary_conditions,
                                              self.iter_args, self.arg_names)
        terminal_boundary_conditions = ca_wrap('Psi_f', self.args, self.source_bvp.compute_terminal_boundary_conditions,
                                               self.iter_args, self.arg_names)
        return initial_boundary_conditions, terminal_boundary_conditions
