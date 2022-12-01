from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.problems.bvp import AdiffInputBVP
from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import SymMatrix
from .compiled import CompBVP
from .symbolic import SymBVP
from ..components.adiff import AdiffBoundaryConditions, ca_wrap


class AdiffBVP(Picky):
    SUPPORTED_INPUTS: type = Union[AdiffInputBVP, SymBVP, CompBVP]

    def __init__(self, source_bvp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_bvp)

        self.src_bvp = deepcopy(source_bvp)

        self.arg_names = ('t', 'x', 'p', 'k')

        if isinstance(self.src_bvp, AdiffInputBVP):
            self.comp_bvp = None

            self.independent = self.src_bvp.independent.values
            self.states = self.src_bvp.states.states
            self.parameters = self.src_bvp.parameters.values
            self.constants = self.src_bvp.constants.constants
            self.default_values = self.src_bvp.constants.default_values
            self.eom = self.src_bvp.states.eoms
            self.inputConstraints = self.src_bvp.constraints

            self.num_states = self.states.numel()
            self.num_parameters = self.parameters.numel()
            self.num_constants = self.constants.numel()

            self.args = (self.independent, self.states, self.parameters, self.constants)
            self.ca_dynamics = ca.Function('f', self.args, (self.eom,), self.arg_names, 'dx_dt')
            self.ca_boundary_conditions = self.create_boundary_conditions()

            self.upper_bounds = {
                't': self.src_ocp.independent.upper_bound,
                'x': self.src_ocp.states.upper_bound,
                'p': self.src_ocp.parameters.upper_bound
            }

            self.lower_bounds = {
                't': self.src_ocp.independent.lower_bound,
                'x': self.src_ocp.states.lower_bound,
                'p': self.src_ocp.parameters.lower_bound
            }

            self.bounding_funcs = {
                't':
                    ca.Function('t_bnd', self.args, (
                        ca.if_else(
                            self.independent < self.lower_bounds['t'], self.lower_bounds['t'],
                            ca.if_else(self.independent > self.upper_bounds['t'], self.upper_bounds['t'],
                                       self.independent)
                        )
                        ,), self.arg_names, ('t_bnd',)),
                'x':
                    ca.Function('t_bnd', self.args, (
                        ca.if_else(
                            self.independent < self.lower_bounds['x'], self.lower_bounds['x'],
                            ca.if_else(self.independent > self.upper_bounds['x'], self.upper_bounds['x'],
                                       self.independent)
                        )
                        ,), self.arg_names, ('t_bnd',)),
                'p':
                    ca.Function('t_bnd', self.args, (
                        ca.if_else(
                            self.independent < self.lower_bounds['p'], self.lower_bounds['p'],
                            ca.if_else(self.independent > self.upper_bounds['p'], self.upper_bounds['p'],
                                       self.independent)
                        )
                        ,), self.arg_names, ('t_bnd',)),
            }

            self.bounded = self.src_ocp.independent.bounded \
                           or self.src_ocp.states.bounded \
                           or self.src_ocp.controls.bounded \
                           or self.src_ocp.parameters.bounded

        else:
            if isinstance(self.src_bvp, CompBVP):
                if self.src_bvp.use_jit_compile:
                    warn('AdiffDual cannot accept JIT compiled CompDual! Recompiling CompDual without JIT...')
                    self.comp_bvp: CompBVP = CompBVP(self.src_bvp.src_bvp, use_jit_compile=False)
                else:
                    self.comp_bvp: CompBVP = deepcopy(self.src_bvp)
                self.constants: SymMatrix = self.src_bvp.src_bvp.constants
            elif isinstance(self.src_bvp, SymBVP):
                self.comp_bvp: CompBVP = CompBVP(self.src_bvp, use_jit_compile=False)
                self.constants: SymMatrix = self.src_bvp.constants

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

            # TODO Implement bounding for symbolic input
            self.upper_bounds = {
                't': None,
                'x': None,
                'p': None
            }

            self.lower_bounds = {
                't': None,
                'x': None,
                'p': None
            }

            self.bounding_funcs = {
                't': None,
                'x': None,
                'p': None
            }

            self.bounded = None

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
