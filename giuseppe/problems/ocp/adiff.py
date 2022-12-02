from copy import deepcopy
from typing import Union
from warnings import warn

import casadi as ca

from giuseppe.utils.mixins import Picky
from giuseppe.utils.typing import SymMatrix
from giuseppe.problems.ocp.adiffInput import AdiffInputOCP
from giuseppe.problems.components.adiffInput import InputAdiffCost, InputAdiffConstraints,\
    InputAdiffInequalityConstraints
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
            self.dtype = self.src_ocp.dtype
            self.comp_ocp = None
            self.independent = self.src_ocp.independent.values
            self.states = self.src_ocp.states.states
            self.controls = self.src_ocp.controls.values
            self.parameters = self.src_ocp.parameters.values
            self.constants = self.src_ocp.constants.constants

            self.default_values = self.src_ocp.constants.default_values
            self.eom = self.src_ocp.states.eoms
            self.inputConstraints = self.src_ocp.constraints
            self.inputCost = self.src_ocp.cost

            self.unregulated_controls = self.controls
            self.ca_pseudo2control = ca.Function('u', (self.controls, self.constants),
                                                 (self.controls,), ('u_reg', 'k'), ('u',))
            self.ca_control2pseudo = ca.Function('u', (self.unregulated_controls, self.constants),
                                                 (self.unregulated_controls,), ('u', 'k'), ('u_reg',))
            self.process_inequality_constraints(self.src_ocp.inequality_constraints)

            self.num_states = self.states.shape[0]
            self.num_controls = self.controls.shape[0]
            self.num_parameters = self.parameters.shape[0]
            self.num_constants = self.constants.shape[0]

            self.args = (self.independent, self.states, self.controls, self.parameters, self.constants)
            self.ca_dynamics = ca.Function('f', self.args, (self.eom,), self.arg_names, ('dx_dt',))
            self.ca_boundary_conditions = self.create_boundary_conditions()
            self.ca_cost = self.create_cost()

            self.upper_bounds = {
                't': self.src_ocp.independent.upper_bound,
                'x': self.src_ocp.states.upper_bound,
                'u': self.src_ocp.controls.upper_bound,
                'p': self.src_ocp.parameters.upper_bound
            }

            self.lower_bounds = {
                't': self.src_ocp.independent.lower_bound,
                'x': self.src_ocp.states.lower_bound,
                'u': self.src_ocp.controls.lower_bound,
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
                    ca.Function('x_bnd', self.args, (
                        ca.if_else(
                            self.states < self.lower_bounds['x'], self.lower_bounds['x'],
                            ca.if_else(self.states > self.upper_bounds['x'], self.upper_bounds['x'],
                                       self.states)
                        )
                        ,), self.arg_names, ('t_bnd',)),
                'u':
                    ca.Function('u_bnd', self.args, (
                        ca.if_else(
                            self.controls < self.lower_bounds['u'], self.lower_bounds['u'],
                            ca.if_else(self.controls > self.upper_bounds['u'], self.upper_bounds['u'],
                                       self.controls)
                        )
                        ,), self.arg_names, ('t_bnd',)),
                'p':
                    ca.Function('p_bnd', self.args, (
                        ca.if_else(
                            self.parameters < self.lower_bounds['p'], self.lower_bounds['p'],
                            ca.if_else(self.parameters > self.upper_bounds['p'], self.upper_bounds['p'],
                                       self.parameters)
                        )
                        ,), self.arg_names, ('t_bnd',)),
            }

            self.bounded = self.src_ocp.independent.bounded \
                           or self.src_ocp.states.bounded \
                           or self.src_ocp.controls.bounded \
                           or self.src_ocp.parameters.bounded

        else:
            self.dtype = ca.SX
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

            # Convert sympy symbolic args to CasADi symbolic args
            self.independent = self.sym2ca_sym(self.src_ocp.sym_args[0])
            self.states = self.sym2ca_sym(self.src_ocp.sym_args[1])
            self.controls = self.sym2ca_sym(self.src_ocp.sym_args[2])
            self.parameters = self.sym2ca_sym(self.src_ocp.sym_args[3])
            self.constants = self.sym2ca_sym(self.src_ocp.sym_args[4])

            self.args = (self.independent, self.states, self.controls, self.parameters, self.constants)
            self.iter_args = [ca.vertsplit(arg, 1) for arg in self.args[1:]]
            self.iter_args.insert(0, self.args[0])  # Insert time separately b/c not wrapped in list

            self.ca_dynamics, self.eom = self.wrap_dynamics()
            self.ca_boundary_conditions, self.inputConstraints = self.wrap_boundary_conditions()
            self.ca_cost, self.inputCost = self.wrap_cost()

            # TODO Implement bounding for symbolic input
            self.upper_bounds = {
                't': None,
                'x': None,
                'u': None,
                'p': None
            }

            self.lower_bounds = {
                't': None,
                'x': None,
                'u': None,
                'p': None
            }

            self.bounding_funcs = {
                't': None,
                'x': None,
                'u': None,
                'p': None
            }

            self.bounded = False

    def sym2ca_sym(self, sympy_sym):
        """

        Parameters
        ----------
        sympy_sym
            A sympy scalar or vector

        Returns
        -------
        ca_sym
            A CasADi symbolic vector

        """
        if hasattr(sympy_sym, '__len__'):
            length = len(sympy_sym)
        else:
            length = 1

        return self.dtype.sym(str(sympy_sym), length)

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

    def process_inequality_constraints(self, input_inequality_constraints: InputAdiffInequalityConstraints):
        for position in ['initial', 'path', 'terminal', 'control']:
            for constraint in input_inequality_constraints.__getattribute__(position):
                if constraint.regularizer is None:
                    raise NotImplementedError('Inequality constraint without regularizer not yet implemented')
                else:
                    constraint.regularizer.apply(self, constraint, position)
