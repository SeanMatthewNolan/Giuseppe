from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.utils.mixins import Picky
from .symbolic import SymDual, SymDualOCP, SymOCP, AlgebraicControlHandler, DifferentialControlHandler
from .compiled import CompDual, CompDualOCP
from ..components.adiff import AdiffBoundaryConditions, AdiffCost, ca_wrap
from ..ocp.compiled import CompCost, CompOCP
from ..dual.compiled import CompAlgControlHandler, CompDiffControlHandler


class AdiffDual(Picky):
    SUPPORTED_INPUTS: type = Union[SymDual, CompDual]  # TODO change to take in SymOCP, CompOCP

    def __init__(self, source_dual: SUPPORTED_INPUTS):
        Picky.__init__(self, source_dual)

        self.src_dual = deepcopy(source_dual)
        self.src_ocp = self.src_dual.src_ocp

        if isinstance(self.src_dual, CompDual):
            if self.src_dual.use_jit_compile:
                warn('AdiffDual cannot accept JIT compiled CompDual! Recompiling CompDual without JIT...')
                self.comp_dual: CompDual = CompDual(self.src_dual.src_dual, use_jit_compile=False)
            else:
                self.comp_dual: CompDual = deepcopy(self.src_dual)
        else:
            self.comp_dual: CompDual = CompDual(self.src_dual, use_jit_compile=False)

        self.comp_ocp: CompOCP = CompOCP(self.src_ocp, use_jit_compile=False)

        self.num_states = self.comp_ocp.num_states
        self.num_costates = self.comp_dual.num_costates
        self.num_controls = self.comp_ocp.num_controls
        self.num_parameters = self.comp_ocp.num_parameters
        self.num_initial_adjoints = self.comp_dual.num_initial_adjoints
        self.num_terminal_adjoints = self.comp_dual.num_terminal_adjoints
        self.num_constants = self.comp_ocp.num_constants

        arg_lens = {'initial': (1, self.num_states, self.num_costates, self.num_controls,
                                self.num_parameters, self.num_initial_adjoints, self.num_constants),
                    'dynamic': (1, self.num_states, self.num_costates, self.num_controls,
                                self.num_parameters, self.num_constants),
                    'terminal': (1, self.num_states, self.num_costates, self.num_controls,
                                 self.num_parameters, self.num_terminal_adjoints, self.num_constants),
                    'ocp': (1, self.num_states, self.num_controls, self.num_parameters, self.num_constants)}

        self.arg_names = {'initial': ('t', 'x', 'lam', 'u', 'p', 'nu_0', 'k'),
                          'dynamic': ('t', 'x', 'lam', 'u', 'p', 'k'),
                          'terminal': ('t', 'x', 'lam', 'u', 'p', 'nu_f', 'k'),
                          'ocp': ('t', 'x', 'u', 'p', 'k')}

        self.args = {}
        self.iter_args = {}
        for key in self.arg_names:
            self.args[key] = [ca.SX.sym(name, length) for name, length in zip(self.arg_names[key], arg_lens[key])]
            self.iter_args[key] = [ca.vertsplit(arg, 1) for arg in self.args[key]]
            self.iter_args[key][0] = self.iter_args[key][0][0]  # TODO 't' not treated as list, but as primitive. How can this be more elegant?


        # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
        self.ca_dynamics = self.wrap_dynamics()
        self.ca_boundary_conditions = self.wrap_boundary_conditions()
        self.ca_cost = self.wrap_cost()

        self.ca_costate_dynamics = self.wrap_costate_dynamics()
        self.ca_adjoined_boundary_conditions = self.wrap_adjoined_boundary_conditions()
        self.ca_augmented_cost = self.wrap_augmented_cost()
        self.ca_hamiltonian = self.ca_augmented_cost.path

    # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
    def wrap_dynamics(self):
        dynamics = ca_wrap('f', self.args['ocp'], self.comp_ocp.dynamics, self.iter_args['ocp'],
                           self.arg_names['ocp'], 'dx_dt')
        return dynamics

    # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
    def wrap_boundary_conditions(self):
        initial_boundary_conditions = ca_wrap('Psi_0', self.args['ocp'], self.comp_ocp.boundary_conditions.initial,
                                              self.iter_args['ocp'], self.arg_names['ocp'])
        terminal_boundary_conditions = ca_wrap('Psi_f', self.args['ocp'], self.comp_ocp.boundary_conditions.terminal,
                                               self.iter_args['ocp'], self.arg_names['ocp'])
        return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
    def wrap_cost(self):
        initial_cost = ca_wrap('Phi_0', self.args['ocp'], self.comp_ocp.cost.initial,
                               self.iter_args['ocp'], self.arg_names['ocp'])
        path_cost = ca_wrap('L', self.args['ocp'], self.comp_ocp.cost.path,
                            self.iter_args['ocp'], self.arg_names['ocp'])
        terminal_cost = ca_wrap('Phi_f', self.args['ocp'], self.comp_ocp.cost.terminal,
                                self.iter_args['ocp'], self.arg_names['ocp'])

        return AdiffCost(initial_cost, path_cost, terminal_cost)

    def wrap_costate_dynamics(self):
        costate_dynamics = ca_wrap('dlam_dt', self.args['dynamic'], self.comp_dual.costate_dynamics,
                                   self.iter_args['dynamic'], self.arg_names['dynamic'])
        return costate_dynamics

    def wrap_adjoined_boundary_conditions(self):
        initial_boundary_conditions = ca_wrap('Psi_0adj', self.args['initial'],
                                              self.comp_dual.adjoined_boundary_conditions.initial,
                                              self.iter_args['initial'], self.arg_names['initial'])
        terminal_boundary_conditions = ca_wrap('Psi_fadj', self.args['terminal'],
                                               self.comp_dual.adjoined_boundary_conditions.terminal,
                                               self.iter_args['terminal'], self.arg_names['terminal'])

        return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)


class AdiffAlgControlHandler:
    def __init__(self, source_handler: AlgebraicControlHandler, adiff_dual: AdiffDual):
        self.src_handler = deepcopy(source_handler)
        self.adiff_dual = adiff_dual

        self.comp_control = self.adiff_dual.comp_dual.control_handler.control
        self.args = self.adiff_dual.args['dynamic']
        self.iter_args = self.adiff_dual.iter_args['dynamic']
        self.arg_names = self.adiff_dual.arg_names['dynamic']

        self.ca_control = self.wrap_control()


    def wrap_control(self):
        return ca_wrap('u', self.args, self.comp_control, self.iter_args, self.arg_names)


# TODO convert CompDiffControlHandler to AdiffDiffControlHandler -wlevin 4/8/2022
class AdiffDiffControlHandler:
    def __init__(self, source_handler: DifferentialControlHandler, adiff_dual: AdiffDual):
        self.src_handler = deepcopy(source_handler)
        self.adiff_dual = adiff_dual

        self.comp_control_dynamics = self.adiff_dual.comp_dual.control_handler.control_dynamics
        self.comp_control_bc = self.adiff_dual.comp_dual.control_handler.control_bc
        self.args = self.adiff_dual.args['dynamic']
        self.iter_args = self.adiff_dual.iter_args['dynamic']
        self.arg_names = self.adiff_dual.arg_names['dynamic']

        self.ca_control_dynamics = self.wrap_control_dynamics()
        self.ca_control_bc = self.wrap_control_bc()


    def wrap_control_dynamics(self):
        return ca_wrap('u_dot', self.args, self.comp_control_dynamics, self.iter_args, self.arg_names)


    def compile_control_bc(self):
        return ca_wrap('Hu_f', self.comp_control_bc, self.args, self.iter_args, self.arg_names)


class AdiffDualOCP(Picky):
    SUPPORTED_INPUTS: type = Union[SymDualOCP, CompDualOCP]

    def __init__(self, source_dualocp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_dualocp)

        self.src_dualocp = deepcopy(source_dualocp)

        if isinstance(self.src_dualocp, CompDualOCP):
            if self.src_dualocp.use_jit_compile:
                warn('AdiffDualOCP cannot accept JIT compiled CompDualOCP! Recompiling CompDualOCP without JIT...')
                self.comp_dualocp: CompDualOCP = CompDualOCP(self.src_dualocp.src_dualocp, use_jit_compile=False)
            else:
                self.comp_dualocp: CompDualOCP = deepcopy(self.src_dualocp)
        else:
            self.comp_dualocp: CompDualOCP = CompDualOCP(self.src_dualocp, use_jit_compile=False)

        self.comp_ocp: CompOCP = self.comp_dualocp.comp_ocp
        self.adiff_dual: AdiffDual = AdiffDual(self.comp_dualocp.comp_dual)
        self.control_handler: Union[AdiffAlgControlHandler, AdiffDiffControlHandler] = self.compile_control_handler()

    def compile_control_handler(self):
        comp_control_handler = self.comp_dualocp.control_handler
        if type(comp_control_handler) is AlgebraicControlHandler:
            return AdiffAlgControlHandler(comp_control_handler, self.adiff_dual)

        elif type(comp_control_handler) is DifferentialControlHandler:
            return AdiffDiffControlHandler(comp_control_handler, self.adiff_dual)
