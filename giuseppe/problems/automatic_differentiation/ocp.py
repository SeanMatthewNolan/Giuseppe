from copy import deepcopy
from typing import Union
from warnings import warn

import numpy as np
import casadi as ca
from numba.core.registry import CPUDispatcher

from giuseppe.data_classes.annotations import Annotations
from giuseppe.problems.protocols import OCP, VectorizedOCP

from .input import ADiffInputProb, ADiffInputInequalityConstraints
from .utils import ca_wrap, lambdify_ca
from ..symbolic.ocp import SymOCP, StrInputProb


class ADiffOCP(VectorizedOCP):
    def __init__(self, source_ocp: Union[ADiffInputProb, SymOCP, OCP, StrInputProb]):
        self.source_ocp = deepcopy(source_ocp)

        self.dyn_arg_names = ('t', 'x', 'u', 'p', 'k')
        self.bc_arg_names = ('t', 'x', 'p', 'k')

        if isinstance(self.source_ocp, ADiffInputProb):
            self.dtype = self.source_ocp.dtype

            self.independent = self.source_ocp.independent
            self.states = self.source_ocp.states.states
            self.controls = self.source_ocp.controls
            self.parameters = self.source_ocp.parameters
            self.constants = self.source_ocp.constants.constants
            self.default_values = self.source_ocp.constants.default_values
            self.eom = self.source_ocp.states.eoms
            self.input_constraints = self.source_ocp.constraints
            self.input_cost = self.source_ocp.cost

            self.annotations: Annotations = self.source_ocp.create_annotations()

            self.num_states = self.states.numel()
            self.num_controls = self.controls.numel()
            self.num_parameters = self.parameters.numel()
            self.num_constants = self.constants.numel()

            self.unregulated_controls = self.controls
            self.ca_pseudo2control = ca.Function(
                    'u', (self.controls, self.constants),
                    (self.controls,), ('u_reg', 'k'), ('u',))
            self.ca_control2pseudo = ca.Function(
                    'u', (self.unregulated_controls, self.constants),
                    (self.unregulated_controls,), ('u', 'k'), ('u_reg',))
            self.process_inequality_constraints(self.source_ocp.inequality_constraints)

            self.dyn_args = (self.independent, self.states, self.controls, self.parameters, self.constants)
            self.bc_args = (self.independent, self.states, self.parameters, self.constants)

            self.ca_dynamics = ca.Function('f', self.dyn_args, (self.eom,), self.dyn_arg_names, ('dx_dt',))
            self.ca_initial_boundary_conditions, self.ca_terminal_boundary_conditions \
                = self.create_boundary_conditions()
            self.ca_initial_cost, self.ca_path_cost, self.ca_terminal_cost = self.create_cost()

        elif isinstance(self.source_ocp, (OCP, SymOCP, StrInputProb)):
            self.dtype = ca.SX

            if isinstance(self.source_ocp, StrInputProb):
                self.source_ocp = SymOCP(self.source_ocp)

            if isinstance(self.source_ocp, SymOCP):
                self.source_ocp = self.source_ocp.compile(use_jit_compile=False)

            if isinstance(self.source_ocp.compute_dynamics, CPUDispatcher) \
                    or isinstance(self.source_ocp.compute_boundary_conditions, CPUDispatcher):
                warn('ADiffBVP cannot accept JIT compiled BVP! Please don\'t JIT compile in this case')

            self.annotations = self.source_ocp.annotations

            if isinstance(self.source_ocp.compute_dynamics, CPUDispatcher):
                self.source_ocp.compute_dynamics = self.source_ocp.compute_dynamics.py_func

            if isinstance(self.source_ocp.compute_boundary_conditions, CPUDispatcher):
                self.source_ocp.compute_boundary_conditions = self.source_ocp.compute_boundary_conditions.py_func

            self.num_states = self.source_ocp.num_states
            self.num_controls = self.source_ocp.num_controls
            self.num_parameters = self.source_ocp.num_parameters
            self.num_constants = self.source_ocp.num_constants
            self.default_values = self.source_ocp.default_values

            self.independent = self.dtype.sym(self.annotations.independent, 1)
            self.states = self.dtype.sym(str(self.annotations.states), self.num_states)
            self.controls = self.dtype.sym(str(self.annotations.controls), self.num_controls)
            self.parameters = self.dtype.sym(str(self.annotations.parameters), self.num_parameters)
            self.constants = self.dtype.sym(str(self.annotations.constants), self.num_constants)

            self.dyn_args = (self.independent, self.states, self.controls, self.parameters, self.constants)
            self.iter_dyn_args = [ca.vertsplit(arg, 1) for arg in self.dyn_args[1:]]
            self.iter_dyn_args.insert(0, self.dyn_args[0])  # Insert time separately b/c not wrapped in list

            self.bc_args = (self.independent, self.states, self.parameters, self.constants)
            self.iter_bc_args = [ca.vertsplit(arg, 1) for arg in self.bc_args[1:]]
            self.iter_bc_args.insert(0, self.bc_args[0])  # Insert time separately b/c not wrapped in list

            self.ca_dynamics = self.wrap_dynamics()

            self.ca_initial_boundary_conditions, self.ca_terminal_boundary_conditions \
                = self.wrap_boundary_conditions()

            self.ca_initial_cost, self.ca_path_cost, self.ca_terminal_cost = self.wrap_cost()

        else:
            raise ValueError('Need a source BVP')

        self.compute_dynamics = lambdify_ca(self.ca_dynamics)

        self.compute_initial_boundary_conditions = lambdify_ca(self.ca_initial_boundary_conditions)
        self.compute_terminal_boundary_conditions = lambdify_ca(self.ca_terminal_boundary_conditions)

        self.compute_initial_cost = lambdify_ca(self.ca_initial_cost)
        self.compute_path_cost = lambdify_ca(self.ca_path_cost)
        self.compute_terminal_cost = lambdify_ca(self.ca_terminal_cost)

        self.compute_dynamics_vectorized, self.compute_path_cost_vectorized = self.vectorize()

    def wrap_dynamics(self):
        return ca_wrap('f', self.dyn_args, self.source_ocp.compute_dynamics,
                       self.iter_dyn_args, self.dyn_arg_names, 'dx_dt')

    def create_boundary_conditions(self):
        initial_boundary_conditions = ca.Function(
                'Psi_0', self.bc_args, (self.input_constraints.initial,), self.bc_arg_names, ('Psi_0',))
        terminal_boundary_conditions = ca.Function(
                'Psi_f', self.bc_args, (self.input_constraints.terminal,), self.bc_arg_names, ('Psi_f',))
        return initial_boundary_conditions, terminal_boundary_conditions

    def wrap_boundary_conditions(self):
        initial_boundary_conditions = ca_wrap(
                'Psi_0', self.bc_args, self.source_ocp.compute_initial_boundary_conditions,
                self.iter_bc_args, self.bc_arg_names)
        terminal_boundary_conditions = ca_wrap(
                'Psi_f', self.bc_args, self.source_ocp.compute_terminal_boundary_conditions,
                self.iter_bc_args, self.bc_arg_names)
        return initial_boundary_conditions, terminal_boundary_conditions

    def wrap_cost(self):
        initial_cost = ca_wrap('Phi_0', self.bc_args, self.source_ocp.compute_initial_cost,
                               self.iter_bc_args, self.bc_arg_names)
        path_cost = ca_wrap('L', self.dyn_args, self.source_ocp.compute_path_cost,
                            self.iter_dyn_args, self.dyn_arg_names)
        terminal_cost = ca_wrap('Phi_f', self.bc_args, self.source_ocp.compute_terminal_cost,
                                self.iter_bc_args, self.bc_arg_names)
        return initial_cost, path_cost, terminal_cost

    def create_cost(self):
        initial_cost = ca.Function('Phi_0', self.bc_args, (self.input_cost.initial,),
                                   self.bc_arg_names, ('Phi_0',))
        path_cost = ca.Function('L', self.dyn_args, (self.input_cost.path,),
                                self.dyn_arg_names, ('L',))
        terminal_cost = ca.Function('Phi_f', self.bc_args, (self.input_cost.terminal,),
                                    self.bc_arg_names, ('Phi_f',))
        return initial_cost, path_cost, terminal_cost

    def process_inequality_constraints(self, input_inequality_constraints: ADiffInputInequalityConstraints):
        for position in ['initial', 'path', 'terminal', 'control']:
            for constraint in input_inequality_constraints.__getattribute__(position):
                if constraint.regularizer is None:
                    raise NotImplementedError('Inequality constraint without regularizer not yet implemented')
                else:
                    constraint.regularizer.apply(self, constraint, position)

    def vectorize(self):
        _compute_dynamics = self.ca_dynamics

        def _compute_dynamics_vectorized(
                independent: np.ndarray, states: np.ndarray, costates: np.ndarray, parameters: np.ndarray,
                constants: np.ndarray
        ) -> np.ndarray:
            map_size = len(independent)
            _dynamics_mapped = _compute_dynamics.map(map_size)

            return np.array(_dynamics_mapped(independent, states, costates, parameters, constants))

        _compute_path_cost = self.ca_path_cost

        def _compute_path_cost_vectorized(
                independent: np.ndarray, states: np.ndarray, costates: np.ndarray, parameters: np.ndarray,
                constants: np.ndarray
        ) -> np.ndarray:
            map_size = len(independent)
            _path_cost_mapped = _compute_path_cost.map(map_size)

            return np.array(_path_cost_mapped(independent, states, costates, parameters, constants))

        return _compute_dynamics_vectorized, _compute_path_cost_vectorized
