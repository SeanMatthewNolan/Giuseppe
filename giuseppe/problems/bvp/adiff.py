from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import SymMatrix
from .compiled import CompBVP
from .symbolic import SymBVP
from ..components.adiff import AdiffBoundaryConditions, ca_wrap


class AdiffBVP(Picky):
    SUPPORTED_INPUTS: type = Union[SymBVP, CompBVP]

    def __init__(self, source_bvp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_bvp)

        self.src_bvp = deepcopy(source_bvp)

        if isinstance(self.src_bvp, CompBVP):
            if self.src_bvp.use_jit_compile:
                warn('AdiffDual cannot accept JIT compiled CompDual! Recompiling CompDual without JIT...')
                self.comp_bvp: CompBVP = CompBVP(self.src_bvp.src_bvp, use_jit_compile=False)
            else:
                self.comp_bvp: CompBVP = deepcopy(self.src_bvp)
            self.constants: SymMatrix = self.src_bvp.src_bvp.constants
        else:
            self.comp_bvp: CompBVP = CompBVP(self.src_bvp, use_jit_compile=False)
            self.constants: SymMatrix = self.src_bvp.constants

        self.num_states = self.comp_bvp.num_states
        self.num_parameters = self.comp_bvp.num_parameters
        self.num_constants = self.comp_bvp.num_constants
        self.default_values = self.comp_bvp.default_values

        arg_lens = (1, self.num_states, self.num_parameters, self.num_constants)
        self.arg_names = ('t', 'x', 'p', 'k')

        self.args = [ca.SX.sym(name, length) for name, length in zip(self.arg_names, arg_lens)]
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
    