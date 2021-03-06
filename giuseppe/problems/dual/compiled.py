from copy import deepcopy
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.utils.complilation import lambdify, jit_compile
from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import NumbaFloat, NumbaArray, SymMatrix
from .symbolic import SymDual, SymDualOCP, SymOCP, AlgebraicControlHandler, DifferentialControlHandler,\
    DifferentialControlHandlerNumeric
from ..components.compiled import CompBoundaryConditions, CompCost
from ..ocp.compiled import CompOCP


class CompDual(Picky):
    SUPPORTED_INPUTS: type = Union[SymDual]

    def __init__(self, source_dual: SUPPORTED_INPUTS, use_jit_compile=True):
        Picky.__init__(self, source_dual)

        self.use_jit_compile = use_jit_compile
        self.src_dual: Union[SymDual] = deepcopy(source_dual)
        self.src_ocp: Union[SymOCP] = self.src_dual.src_ocp

        self.num_costates = len(self.src_dual.costates)
        self.num_initial_adjoints = len(self.src_dual.initial_adjoints)
        self.num_terminal_adjoints = len(self.src_dual.terminal_adjoints)

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

        self.costate_dynamics = self.compile_costate_dynamics()
        self.adjoined_boundary_conditions = self.compile_adjoined_boundary_conditions()
        self.augmented_cost = self.compile_augemented_cost()
        self.hamiltonian = self.augmented_cost.path

    def compile_costate_dynamics(self):
        return lambdify(self.sym_args['dynamic'], self.src_dual.costate_dynamics.flat(),
                        use_jit_compile=self.use_jit_compile)

    def compile_adjoined_boundary_conditions(self):
        initial_boundary_conditions = lambdify(
                self.sym_args['initial'], self.src_dual.adjoined_boundary_conditions.initial.flat(),
                use_jit_compile=self.use_jit_compile)
        terminal_boundary_conditions = lambdify(
                self.sym_args['terminal'], self.src_dual.adjoined_boundary_conditions.terminal.flat(),
                use_jit_compile=self.use_jit_compile)

        return CompBoundaryConditions(initial_boundary_conditions, terminal_boundary_conditions)

    def compile_augemented_cost(self):
        initial_aug_cost = lambdify(self.sym_args['initial'], self.src_dual.augmented_cost.initial,
                                    use_jit_compile=self.use_jit_compile)
        hamiltonian = lambdify(self.sym_args['dynamic'], self.src_dual.augmented_cost.path,
                               use_jit_compile=self.use_jit_compile)
        terminal_aug_cost = lambdify(self.sym_args['terminal'], self.src_dual.augmented_cost.terminal,
                                     use_jit_compile=self.use_jit_compile)

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
    def __init__(self, source_handler: DifferentialControlHandler, comp_dual: CompDual):
        self.src_handler = deepcopy(source_handler)
        self.comp_dual = comp_dual

        self.use_jit_compile = comp_dual.use_jit_compile
        self.sym_args = self.comp_dual.sym_args['dynamic']
        self.args_numba_signature = self.comp_dual.args_numba_signature['dynamic']

        self.control_dynamics = self.compile_control_rate()
        self.control_bc = self.compile_control_bc()

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
    SUPPORTED_INPUTS: type = Union[SymDualOCP]

    def __init__(self, source_dualocp: SUPPORTED_INPUTS, use_jit_compile=True):
        Picky.__init__(self, source_dualocp)

        self.src_dualocp: SymDualOCP = deepcopy(source_dualocp)
        self.use_jit_compile = use_jit_compile

        self.comp_ocp: CompOCP = CompOCP(self.src_dualocp.ocp, use_jit_compile=self.use_jit_compile)
        self.comp_dual: CompDual = CompDual(self.src_dualocp.dual, use_jit_compile=self.use_jit_compile)
        self.control_handler: Union[CompAlgControlHandler, CompDiffControlHandler] = self.compile_control_handler()

    def compile_control_handler(self):
        sym_control_handler = self.src_dualocp.control_handler
        if type(sym_control_handler) is AlgebraicControlHandler:
            return CompAlgControlHandler(sym_control_handler, self.comp_dual)

        elif type(sym_control_handler) is DifferentialControlHandler:
            return CompDiffControlHandler(sym_control_handler, self.comp_dual)

        elif type(sym_control_handler) is DifferentialControlHandlerNumeric:
            return CompDiffControlHandlerNumeric(sym_control_handler, self.comp_dual)
