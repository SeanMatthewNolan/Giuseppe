from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import SymMatrix
from .symbolic import SymOCP
from .compiled import CompOCP
from ..components.adiff import AdiffBoundaryConditions, AdiffCost, ca_wrap


class AdiffOCP(Picky):
    SUPPORTED_INPUTS: type = Union[SymOCP, CompOCP]

    def __init__(self, source_ocp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_ocp)
        self.src_ocp = deepcopy(source_ocp)

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

        arg_lens = (1, self.num_states, self.num_controls, self.num_parameters, self.num_constants)
        self.arg_names = ('t', 'x', 'u', 'p', 'k')

        self.args = [ca.SX.sym(name, length) for name, length in zip(self.arg_names, arg_lens)]
        self.iter_args = [ca.vertsplit(arg, 1) for arg in self.args[1:]]
        self.iter_args.insert(0, self.args[0])  # Insert time separately b/c not wrapped in list

        self.ca_dynamics = self.wrap_dynamics()
        self.ca_boundary_conditions = self.wrap_boundary_conditions()
        self.ca_cost = self.wrap_cost()

    def wrap_dynamics(self):
        dynamics = ca_wrap('f', self.args, self.comp_ocp.dynamics, self.iter_args,
                           self.arg_names, 'dx_dt')
        return dynamics

    # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
    def wrap_boundary_conditions(self):
        initial_boundary_conditions = ca_wrap('Psi_0', self.args, self.comp_ocp.boundary_conditions.initial,
                                              self.iter_args, self.arg_names)
        terminal_boundary_conditions = ca_wrap('Psi_f', self.args, self.comp_ocp.boundary_conditions.terminal,
                                               self.iter_args, self.arg_names)
        return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
    def wrap_cost(self):
        initial_cost = ca_wrap('Phi_0', self.args, self.comp_ocp.cost.initial,
                               self.iter_args, self.arg_names)
        path_cost = ca_wrap('L', self.args, self.comp_ocp.cost.path,
                            self.iter_args, self.arg_names)
        terminal_cost = ca_wrap('Phi_f', self.args, self.comp_ocp.cost.terminal,
                                self.iter_args, self.arg_names)

        return AdiffCost(initial_cost, path_cost, terminal_cost)
