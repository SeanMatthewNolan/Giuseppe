from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import SymMatrix
from giuseppe.problems.ocp.adiffInput import AdiffInputOCP
from giuseppe.problems.components.adiffInput import InputAdiffCost, InputAdiffConstraints
from .compiled import CompOCP
from .symbolic import SymOCP
from ..components.adiff import AdiffBoundaryConditions, AdiffCost, ca_wrap


class AdiffOCP(Picky):
    SUPPORTED_INPUTS: type = Union[AdiffInputOCP, SymOCP, CompOCP]

    def __init__(self, source_ocp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_ocp)
        self.src_ocp = deepcopy(source_ocp)

        self.arg_names = ('t', 'x', 'u', 'p', 'k')

        if isinstance(self.src_ocp, AdiffInputOCP):
            self.comp_ocp = None
            self.independent = self.src_ocp.independent
            self.states = self.src_ocp.states.states
            self.controls = self.src_ocp.controls
            self.parameters = self.src_ocp.parameters
            self.constants = self.src_ocp.constants.constants

            self.default_values = self.src_ocp.constants.default_values
            self.eom = self.src_ocp.states.eoms
            self.inputConstraints = self.src_ocp.constraints
            self.inputCost = self.src_ocp.cost

            self.num_states = self.states.shape[0]
            self.num_controls = self.controls.shape[0]
            self.num_parameters = self.parameters.shape[0]
            self.num_constants = self.constants.shape[0]

            self.args = (self.independent, self.states, self.controls, self.parameters, self.constants)
            self.ca_dynamics = ca.Function('f', self.args, (self.eom,), self.arg_names, ('dx_dt',))
            self.ca_boundary_conditions = self.create_boundary_conditions()
            self.ca_cost = self.create_cost()

        else:
            if isinstance(self.src_ocp, CompOCP):
                if self.src_ocp.use_jit_compile:
                    warn('AdiffDual cannot accept JIT compiled CompDual! Recompiling CompDual without JIT...')
                    self.comp_ocp: CompOCP = CompOCP(self.src_ocp.src_ocp, use_jit_compile=False)
                else:
                    self.comp_ocp: CompOCP = deepcopy(self.src_ocp)
                self.constants: SymMatrix = self.src_ocp.src_ocp.constants
            else:
                self.comp_ocp: CompOCP = CompOCP(self.src_ocp, use_jit_compile=False)
                self.constants: SymMatrix = self.src_ocp.constants

            self.num_states = self.comp_ocp.num_states
            self.num_parameters = self.comp_ocp.num_parameters
            self.num_constants = self.comp_ocp.num_constants
            self.num_controls = self.comp_ocp.num_controls

            self.default_values = self.comp_ocp.default_values

            # TODO constants need to have given string as name, not k_1, k_2, etc.
            self.independent = ca.SX.sym('t', 1)
            self.states = ca.SX.sym('x', self.num_states)
            self.controls = ca.SX.sym('u', self.num_controls)
            self.parameters = ca.SX.sym('p', self.num_parameters)
            self.constants = ca.SX.sym('k', self.num_constants)

            self.args = (self.independent, self.states, self.controls, self.parameters, self.constants)
            self.iter_args = [ca.vertsplit(arg, 1) for arg in self.args[1:]]
            self.iter_args.insert(0, self.args[0])  # Insert time separately b/c not wrapped in list

            self.ca_dynamics, self.eom = self.wrap_dynamics()
            self.ca_boundary_conditions, self.inputConstraints = self.wrap_boundary_conditions()
            self.ca_cost, self.inputCost = self.wrap_cost()

    def wrap_dynamics(self):
        dynamics_fun = ca_wrap('f', self.args, self.comp_ocp.dynamics, self.iter_args,
                               self.arg_names, 'dx_dt')
        dynamics_sym = dynamics_fun(*self.args)
        return dynamics_fun, dynamics_sym

    def wrap_boundary_conditions(self):
        initial_boundary_conditions = ca_wrap('Psi_0', self.args, self.comp_ocp.boundary_conditions.initial,
                                              self.iter_args, self.arg_names)
        terminal_boundary_conditions = ca_wrap('Psi_f', self.args, self.comp_ocp.boundary_conditions.terminal,
                                               self.iter_args, self.arg_names)
        adiff_bcs = AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)
        input_bcs = InputAdiffConstraints(initial=initial_boundary_conditions(*self.args),
                                          terminal=terminal_boundary_conditions(*self.args))
        return adiff_bcs, input_bcs

    def create_boundary_conditions(self):
        initial_boundary_conditions = ca.Function('Psi_0', self.args, (self.inputConstraints.initial,),
                                                  self.arg_names, ('Psi_0',))
        terminal_boundary_conditions = ca.Function('Psi_f', self.args, (self.inputConstraints.terminal,),
                                                   self.arg_names, ('Psi_f',))
        return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    def wrap_cost(self):
        initial_cost = ca_wrap('Phi_0', self.args, self.comp_ocp.cost.initial,
                               self.iter_args, self.arg_names)
        path_cost = ca_wrap('L', self.args, self.comp_ocp.cost.path,
                            self.iter_args, self.arg_names)
        terminal_cost = ca_wrap('Phi_f', self.args, self.comp_ocp.cost.terminal,
                                self.iter_args, self.arg_names)
        adiff_cost = AdiffCost(initial_cost, path_cost, terminal_cost)
        input_cost = InputAdiffCost(initial=initial_cost(*self.args),
                                    path=path_cost(*self.args),
                                    terminal=terminal_cost(*self.args))
        return adiff_cost, input_cost

    def create_cost(self):
        initial_cost = ca.Function('Phi_0', self.args, (self.inputCost.initial,),
                                   self.arg_names, ('Phi_0',))
        path_cost = ca.Function('L', self.args, (self.inputCost.path,),
                                self.arg_names, ('L',))
        terminal_cost = ca.Function('Phi_f', self.args, (self.inputCost.terminal,),
                                    self.arg_names, ('Phi_f',))
        return AdiffCost(initial_cost, path_cost, terminal_cost)
