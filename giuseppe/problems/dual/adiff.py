from collections.abc import Callable
from copy import deepcopy
from typing import Union
from warnings import warn
from dataclasses import dataclass

import numpy as np
import casadi as ca
from numpy.typing import ArrayLike

from giuseppe.utils.complilation import lambdify, jit_compile
from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import NumbaFloat, NumbaArray, SymMatrix
from .symbolic import SymDual, SymDualOCP, SymOCP, AlgebraicControlHandler, DifferentialControlHandler
from .compiled import CompDual, CompDualOCP
from ..components.adiff import AdiffBoundaryConditions, AdiffCost
from ..ocp.compiled import CompCost, CompOCP


class AdiffDual(Picky):
    SUPPORTED_INPUTS: type = Union[SymDual, CompDual]

    def __init__(self, source_dual: SUPPORTED_INPUTS):
        Picky.__init__(self, source_dual)

        self.src_dual = deepcopy(source_dual)
        self.src_ocp = self.src_dual.src_ocp

        if isinstance(self.src_dual, CompDual):
            if self.src_dual.use_jit_compile:
                warn('AdiffDual cannot accept JIT compiled CompDual! Recompiling CompDual without JIT')
                self.comp_dual: CompDual = CompDual(self.src_dual.src_dual, use_jit_compile=False)
            else:
                self.comp_dual: CompDual = deepcopy(self.src_dual)
        else:
            self.comp_dual: CompDual = CompDual(self.src_dual, use_jit_compile=False)

        # if isinstance(source_dualocp, CompDualOCP):
        #     if source_dualocp.use_jit_compile:
        #         warn('AdiffDualOCP cannot accept JIT compiled CompDualOCP! Recompiling CompDualOCP without JIT')
        #         self.comp_dualocp: CompDualOCP = CompDualOCP(source_dualocp.src_dualocp, use_jit_compile=False)
        #     else:
        #         self.comp_dualocp: CompDualOCP = deepcopy(source_dualocp)
        # else:
        #     self.comp_dualocp: CompDualOCP = CompDualOCP(source_dualocp, use_jit_compile=False)

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

        # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
        self.ca_dynamics = self.wrap_dynamics()
        # self.ca_boundary_conditions = self.wrap_boundary_conditions()
        self.ca_cost = self.wrap_cost()

        # TODO boundary_conditions compiled BCs wrap with np.array, can't be wrapped with ca.Function -wlevin 4/8/2022

        self.ca_costate_dynamics = self.wrap_costate_dynamics()
        self.ca_adjoined_boundary_conditions = self.wrap_adjoined_boundary_conditions()

    # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
    def wrap_dynamics(self):
        dynamics = ca.Function('f', self.args['ocp'],
                               (self.comp_ocp.dynamics(*self.iter_args['ocp']),),
                               self.arg_names['ocp'], ('dx_dt',))
        return dynamics

    # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
    def wrap_boundary_conditions(self):
        initial_boundary_conditions = ca.Function('Psi_0', self.args['ocp'],
                                                  (self.comp_ocp.boundary_conditions.initial(*self.iter_args['ocp']),),
                                                  self.arg_names['ocp'], ('Psi_0',))
        terminal_boundary_conditions = ca.Function('Psi_f', self.args['ocp'],
                                                   (self.comp_ocp.boundary_conditions.terminal(*self.iter_args['ocp']),),
                                                   self.arg_names['ocp'], ('Psi_f',))
        return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    # TODO refactor into AdiffOCP class? -wlevin 4/8/2022
    def wrap_cost(self):
        initial_cost = ca.Function('Phi_0', self.args['ocp'],
                                   (self.comp_ocp.cost.initial(*self.iter_args['ocp']),),
                                   self.arg_names['ocp'], ('Phi_0',))
        path_cost = ca.Function('L', self.args['ocp'],
                                (self.comp_ocp.cost.path(*self.iter_args['ocp']),),
                                self.arg_names['ocp'], ('L',))
        terminal_cost = ca.Function('Phi_f', self.args['ocp'],
                                    (self.comp_ocp.cost.terminal(*self.iter_args['ocp']),),
                                    self.arg_names['ocp'], ('Phi_f',))

        return AdiffCost(initial_cost, path_cost, terminal_cost)

    def wrap_costate_dynamics(self):
        costate_dynamics = ca.Function('dlam_dt', self.args['dynamic'],
                                       (self.comp_dual.costate_dynamics(*self.iter_args['dynamic']),),
                                       self.arg_names['dynamic'], ('dlam_dt',))
        return costate_dynamics

    def wrap_adjoined_boundary_conditions(self):
        initial_boundary_conditions = ca.Function('Psi_0adj', self.args['initial'],
                                                  (self.comp_dual.adjoined_boundary_conditions.initial(
                                                      *self.iter_args['initial']),),
                                                  self.arg_names['initial'], ('Psi_0,adj',))
        terminal_boundary_conditions = ca.Function('Psi_fadj', self.args['terminal'],
                                                   (self.comp_dual.adjoined_boundary_conditions.terminal(
                                                       *self.iter_args['terminal']),),
                                                   self.arg_names['terminal'], ('Psi_f,adj',))

        return AdiffBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    # def compile_adjoined_boundary_conditions(self):
    #     return CompBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)
    #
    # def compile_augemented_cost(self):
    #     return CompCost(initial_aug_cost, hamiltonian, terminal_aug_cost)


# TODO convert CompAlgControlHandler to AdiffAlgControlHandler - wlevin 4/8/2022
class AdiffAlgControlHandler:
    def __init__(self, source_handler: AlgebraicControlHandler, comp_dual: CompDual):
        self.src_handler = deepcopy(source_handler)
        self.comp_dual = comp_dual

        self.use_jit_compile = comp_dual.use_jit_compile
        self.sym_args = (self.comp_dual.src_ocp.independent, self.comp_dual.src_ocp.states.flat(),
                         self.comp_dual.src_dual.costates.flat(), self.comp_dual.src_ocp.parameters.flat(),
                         self.comp_dual.src_ocp.constants.flat())
        self.args_numba_signature = (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray)
        self.control = self.compile_control()

    def compile_control(self):
        control_law = self.src_handler.control_law

        num_options = len(control_law)

        if num_options == 1:
            lam_control = lambdify(self.sym_args, list(control_law[-1]), use_jit_compile=self.use_jit_compile)

            def control(t: float, x: ArrayLike, lam: ArrayLike, p: ArrayLike, k: ArrayLike):
                return np.array(lam_control(t, x, lam, p, k))

        else:
            hamiltonian = self.comp_dual.hamiltonian
            lam_control = lambdify(self.sym_args, SymMatrix(control_law), use_jit_compile=self.use_jit_compile)

            def control(t: float, x: ArrayLike, lam: ArrayLike, p: ArrayLike, k: ArrayLike):
                control_options = lam_control(t, x, lam, p, k)
                hamiltonians = np.empty((num_options,))
                for idx in range(num_options):
                    hamiltonians[idx] = hamiltonian(t, x, lam, control_options[idx], p, k)

                return control_options[np.argmin(hamiltonians)]

        if self.use_jit_compile:
            return jit_compile(control, self.args_numba_signature)
        else:
            return control


# TODO convert CompDiffControlHandler to AdiffDiffControlHandler -wlevin 4/8/2022
class AdiffDiffControlHandler:
    def __init__(self, source_handler: DifferentialControlHandler, comp_dual: CompDual):
        self.src_handler = deepcopy(source_handler)
        self.comp_dual = comp_dual

        self.use_jit_compile = comp_dual.use_jit_compile
        self.sym_args = self.comp_dual.sym_args['dynamic']
        self.args_numba_signature = self.comp_dual.args_numba_signature['dynamic']

        self.control_dynamics = self.compile_control_rate()
        self.control_bc = self.compile_control_bc()

    def compile_control_rate(self):
        lam_control_dynamics = lambdify(self.sym_args, self.src_handler.control_dynamics.flat(),
                                        use_jit_compile=self.use_jit_compile)

        def control_dynamics(t: float, x: ArrayLike, lam: ArrayLike, u: ArrayLike, p: ArrayLike, k: ArrayLike):
            return np.array(lam_control_dynamics(t, x, lam, u, p, k))

        if self.use_jit_compile:
            return jit_compile(control_dynamics, self.args_numba_signature)
        else:
            return control_dynamics

    def compile_control_bc(self):
        lam_control_bc = lambdify(self.sym_args, self.src_handler.h_u.flat(), use_jit_compile=self.use_jit_compile)

        def control_bc(t: float, x: ArrayLike, lam: ArrayLike, u: ArrayLike, p: ArrayLike, k: ArrayLike):
            return np.array(lam_control_bc(t, x, lam, u, p, k))

        if self.use_jit_compile:
            return jit_compile(control_bc, self.args_numba_signature)
        else:
            return control_bc


class AdiffDualOCP(Picky):
    SUPPORTED_INPUTS: type = Union[SymDualOCP, CompDualOCP]

    def __init__(self, source_dualocp: SUPPORTED_INPUTS):
        Picky.__init__(self, source_dualocp)

        self.src_dualocp = deepcopy(source_dualocp)

        if isinstance(self.src_dualocp, CompDualOCP):
            if self.src_dualocp.use_jit_compile:
                warn('AdiffDualOCP cannot accept JIT compiled CompDualOCP! Recompiling CompDualOCP without JIT')
                self.comp_dualocp: CompDualOCP = CompDualOCP(self.src_dualocp.src_dualocp, use_jit_compile=False)
            else:
                self.comp_dualocp: CompDualOCP = deepcopy(self.src_dualocp)
        else:
            self.comp_dualocp: CompDualOCP = CompDualOCP(self.src_dualocp, use_jit_compile=False)

        self.comp_ocp: CompOCP = self.comp_dualocp.comp_ocp
        self.adiff_dual: AdiffDual = AdiffDual(self.comp_dualocp.comp_dual)
    #     self.control_handler: Union[AdiffAlgControlHandler, AdiffDiffControlHandler] = self.compile_control_handler()
    #
    # def compile_control_handler(self):
    #     comp_control_handler = self.comp_dualocp.control_handler
    #     if type(comp_control_handler) is AlgebraicControlHandler:
    #         return AdiffAlgControlHandler(comp_control_handler, self.adiff_dual)
    #
    #     elif type(comp_control_handler) is DifferentialControlHandler:
    #         return AdiffDiffControlHandler(comp_control_handler, self.adiff_dual)
