from copy import deepcopy
from typing import Union, Optional
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.utils.compilation import lambdify, jit_compile
from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import NumbaFloat, NumbaArray, SymMatrix
from .symbolic import SymDual, SymDualOCP, SymOCP, AlgebraicControlHandler, DifferentialControlHandler, \
    DifferentialControlHandlerNumeric
from .adiff import AdiffDual, AdiffDualOCP, AdiffOCP, AdiffDiffControlHandler
from ..components.compiled import CompBoundaryConditions, CompCost
from ..components.adiff import lambdify_ca
from ..ocp.compiled import CompOCP


class CompDual(Picky):
    SUPPORTED_INPUTS: type = Union[SymDual, AdiffDual]

    def __init__(self, source_dual: SUPPORTED_INPUTS, use_jit_compile: Optional[bool] = None):
        Picky.__init__(self, source_dual)

        self.src_dual: Union[SymDual, AdiffDual] = deepcopy(source_dual)
        self.src_ocp: Union[SymOCP, CompOCP, AdiffOCP] = self.src_dual.src_ocp

        self.num_costates = self.src_dual.num_costates
        self.num_initial_adjoints = self.src_dual.num_initial_adjoints
        self.num_terminal_adjoints = self.src_dual.num_terminal_adjoints

        if isinstance(self.src_dual, SymDual):
            if use_jit_compile is None:
                self.use_jit_compile = True
            else:
                self.use_jit_compile = use_jit_compile

            self.sym_args = {
                'initial':  (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_dual.costates.flat(),
                             self.src_ocp.controls.flat(), self.src_ocp.parameters.flat(),
                             self.src_dual.initial_adjoints.flat(), self.src_ocp.constants.flat()),
                'dynamic':  (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_dual.costates.flat(),
                             self.src_ocp.controls.flat(), self.src_ocp.parameters.flat(), self.src_ocp.constants.flat()),
                'terminal': (self.src_ocp.independent, self.src_ocp.states.flat(), self.src_dual.costates.flat(),
                             self.src_ocp.controls.flat(), self.src_ocp.parameters.flat(),
                             self.src_dual.terminal_adjoints, self.src_ocp.constants.flat())
            }

            self.args_numba_signature = {
                'initial':  (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
                'dynamic':  (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
                'terminal': (NumbaFloat, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray, NumbaArray),
            }

            self.costate_dynamics = self.compile_sym_costate_dynamics()
            self.adjoined_boundary_conditions = self.compile_sym_adjoined_boundary_conditions()
            self.augmented_cost = self.compile_sym_augmented_cost()
        elif isinstance(self.src_dual, AdiffDual):
            if use_jit_compile is True:
                warn('Cannot JIT compile AdiffDual! Setting use_jit_compile to False')
            self.use_jit_compile = False

            self.sym_args = None
            self.args_numba_signature = None

            self.costate_dynamics = lambdify_ca(self.src_dual.ca_costate_dynamics)
            self.adjoined_boundary_conditions = self.wrap_ca_adj_boundary_conditions()
            self.augmented_cost = self.wrap_ca_augmented_cost()
        else:
            raise TypeError(f"CompDual must be initialized with SymDual or CompDual; you used {type(self.src_dual)}!")

        self.hamiltonian = self.augmented_cost.path

    def compile_sym_costate_dynamics(self):
        return lambdify(self.sym_args['dynamic'], self.src_dual.costate_dynamics.flat(),
                        use_jit_compile=self.use_jit_compile)

    def compile_sym_adjoined_boundary_conditions(self):
        initial_boundary_conditions = lambdify(
                self.sym_args['initial'], self.src_dual.adjoined_boundary_conditions.initial.flat(),
                use_jit_compile=self.use_jit_compile)
        terminal_boundary_conditions = lambdify(
                self.sym_args['terminal'], self.src_dual.adjoined_boundary_conditions.terminal.flat(),
                use_jit_compile=self.use_jit_compile)

        return CompBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    def wrap_ca_adj_boundary_conditions(self):
        initial_boundary_conditions = lambdify_ca(self.src_dual.ca_adj_boundary_conditions.initial)
        terminal_boundary_conditions = lambdify_ca(self.src_dual.ca_adj_boundary_conditions.terminal)

        return CompBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    def compile_sym_augmented_cost(self):
        initial_aug_cost = lambdify(self.sym_args['initial'], self.src_dual.augmented_cost.initial,
                                    use_jit_compile=self.use_jit_compile)
        hamiltonian = lambdify(self.sym_args['dynamic'], self.src_dual.augmented_cost.path,
                               use_jit_compile=self.use_jit_compile)
        terminal_aug_cost = lambdify(self.sym_args['terminal'], self.src_dual.augmented_cost.terminal,
                                     use_jit_compile=self.use_jit_compile)

        return CompCost(initial_aug_cost, hamiltonian, terminal_aug_cost)

    def wrap_ca_augmented_cost(self):
        initial_aug_cost = lambdify_ca(self.src_dual.ca_augmented_cost.initial)
        hamiltonian = lambdify_ca(self.src_dual.ca_augmented_cost.path)
        terminal_aug_cost = lambdify_ca(self.src_dual.ca_augmented_cost.terminal)

        return CompCost(initial_aug_cost, hamiltonian, terminal_aug_cost)


class CompAlgControlHandler:
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
                return np.asarray(lam_control(t, x, lam, p, k))

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


class CompDiffControlHandler:
    def __init__(self, source_handler: Union[DifferentialControlHandler, AdiffDiffControlHandler], comp_dual: CompDual):
        self.src_handler = deepcopy(source_handler)
        self.comp_dual = deepcopy(comp_dual)

        self.use_jit_compile = self.comp_dual.use_jit_compile

        if isinstance(self.src_handler, DifferentialControlHandler):
            self.sym_args = self.comp_dual.sym_args['dynamic']
            self.args_numba_signature = self.comp_dual.args_numba_signature['dynamic']

            self.control_dynamics = self.compile_control_rate()
            self.control_bc = self.compile_control_bc()
        elif isinstance(self.src_handler, AdiffDiffControlHandler):
            self.sym_args = None
            self.args_numba_signature = None

            self.control_dynamics = lambdify_ca(self.src_handler.ca_control_dynamics)
            self.control_bc = lambdify_ca(self.src_handler.ca_control_bc)
        else:
            raise TypeError(f"CompDiffControlHandler must be initialized with DifferentialControlHandler or AdiffDiffControlHandler; you used {type(self.src_handler)}!")

    def compile_control_rate(self):
        return lambdify(self.sym_args, self.src_handler.control_dynamics.flat(), use_jit_compile=self.use_jit_compile)

    def compile_control_bc(self):
        return lambdify(self.sym_args, self.src_handler.h_u.flat(), use_jit_compile=self.use_jit_compile)


class CompDiffControlHandlerNumeric:
    def __init__(self, source_handler: DifferentialControlHandlerNumeric, comp_dual: CompDual):
        self.src_handler = deepcopy(source_handler)
        self.comp_dual = comp_dual

        self.use_jit_compile = comp_dual.use_jit_compile
        self.sym_args = self.comp_dual.sym_args['dynamic']
        self.args_numba_signature = self.comp_dual.args_numba_signature['dynamic']

        self.control_dynamics = self.compile_control_rate()
        self.control_bc = self.compile_control_bc()

    def compile_control_rate(self):
        compute_h_uu = lambdify(self.sym_args, self.src_handler.h_uu, use_jit_compile=self.use_jit_compile)
        compute_rhs = lambdify(self.sym_args, self.src_handler.rhs.flat(), use_jit_compile=self.use_jit_compile)

        def control_dynamics(_t, _x, _lam, _u, _p, _k):
            return np.linalg.solve(-compute_h_uu(_t, _x, _lam, _u, _p, _k),
                                   np.asarray(compute_rhs(_t, _x, _lam, _u, _p, _k)))

        if self.use_jit_compile:
            control_dynamics = jit_compile(control_dynamics, self.args_numba_signature)

        return control_dynamics

    def compile_control_bc(self):
        return lambdify(self.sym_args, self.src_handler.h_u.flat(), use_jit_compile=self.use_jit_compile)


class CompDualOCP(Picky):
    SUPPORTED_INPUTS: type = Union[SymDualOCP, AdiffDualOCP]

    def __init__(self, source_dualocp: SUPPORTED_INPUTS, use_jit_compile: Optional[bool] = None):
        Picky.__init__(self, source_dualocp)

        self.src_dualocp: Union[SymDualOCP, AdiffDualOCP] = deepcopy(source_dualocp)

        if isinstance(self.src_dualocp, SymDualOCP):
            if use_jit_compile is None:
                self.use_jit_compile = True
            else:
                self.use_jit_compile = use_jit_compile
            self.comp_ocp: CompOCP = CompOCP(self.src_dualocp.ocp, use_jit_compile=self.use_jit_compile)
        elif isinstance(self.src_dualocp, AdiffDualOCP):
            if use_jit_compile is True:
                warn('Cannot JIT compile AdiffDual! Setting use_jit_compile to False')
            self.use_jit_compile = False
            self.comp_ocp: CompOCP = self.src_dualocp.ocp.comp_ocp
        else:
            TypeError(f"CompDualOCP must be initialized with SymDualOCP or AdiffDualOCP; you used {type(self.src_dualocp)}!")

        self.comp_dual: CompDual = CompDual(self.src_dualocp.dual, use_jit_compile=self.use_jit_compile)
        self.control_handler: Union[CompAlgControlHandler, CompDiffControlHandler] = self.compile_control_handler()

    def compile_control_handler(self):
        src_control_handler = self.src_dualocp.control_handler
        if type(src_control_handler) is AlgebraicControlHandler:
            return CompAlgControlHandler(src_control_handler, self.comp_dual)

        elif type(src_control_handler) is DifferentialControlHandler or AdiffDiffControlHandler:
            return CompDiffControlHandler(src_control_handler, self.comp_dual)

        elif type(src_control_handler) is DifferentialControlHandlerNumeric:
            return CompDiffControlHandlerNumeric(src_control_handler, self.comp_dual)
