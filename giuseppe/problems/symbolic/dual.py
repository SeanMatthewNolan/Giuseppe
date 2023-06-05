import warnings
from typing import Optional

import numpy as np

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import Dual

from .input import StrInputProb
from .adjoints import SymAdjoints
from .ocp import SymOCP


class SymDual(SymOCP, SymAdjoints, Dual):
    def __init__(
            self, input_data: StrInputProb,
            control_method: Optional[str] = 'differential', use_jit_compile: bool = True
    ):

        super().__init__(input_data, use_jit_compile=use_jit_compile)

        self.ocp_sym_args = self.sym_args

        super()._sympify_adjoint_information(self)
        super()._compile_adjoint_information(self)

        self.prob_class = 'dual'

        self.control_method: Optional[str] = control_method
        if self.control_method is None:
            self.control_handler = None
        elif self.control_method.lower() == 'algebraic':
            from .control_handlers import SymAlgebraicControlHandler
            self.control_handler: SymAlgebraicControlHandler = SymAlgebraicControlHandler(self)
        elif self.control_method.lower() == 'differential':
            from .control_handlers import SymDifferentialControlHandler
            self.control_handler: SymDifferentialControlHandler = SymDifferentialControlHandler(self)
        else:
            raise NotImplementedError(
                    f'\"{control_method}\" is not an implemented control method. Try \"differential\".')

    def add_preprocess(self, name: str):
        pass

    def add_post_process(self, name: str):
        if name == 'compute_hamiltonian':
            def _compute_hamiltonian(_dual: Dual, _data: Solution):
                _data.aux['Hamiltonian'] = np.array(
                        [_dual.compute_hamiltonian(ti, xi, lami, ui, _data.p, _data.k)
                         for ti, xi, lami, ui in zip(_data.t, _data.x.T, _data.lam.T, _data.u.T)])
                return _data

            self.post_processes.append(_compute_hamiltonian)

        elif name == 'compute_huu':
            from .control_handlers import SymDifferentialControlHandler
            from giuseppe.utils.compilation import lambdify
            import sympy

            if isinstance(self.control_handler, SymDifferentialControlHandler):
                _compute_h_uu = self.control_handler.compute_h_uu
            else:
                _h_uu = self.control_law.jacobian(self.controls)
                _compute_h_uu = lambdify(self.sym_args['dynamic'], _h_uu, use_jit_compile=self.use_jit_compile)

            # _h_xx = sympy.Matrix([self.hamiltonian]).jacobian(self.states).jacobian(self.states)
            # _h_ux = self.control_law.jacobian(self.states)
            # _h_ulam = self.control_law.jacobian(self.costates)

            # _compute_h_xx = lambdify(self.sym_args['dynamic'], _h_xx, use_jit_compile=self.use_jit_compile)
            # _compute_h_ux = lambdify(self.sym_args['dynamic'], _h_ux, use_jit_compile=self.use_jit_compile)
            # _compute_h_ulam = lambdify(self.sym_args['dynamic'], _h_ulam, use_jit_compile=self.use_jit_compile)

            def compute_huu(_dual: Dual, _data: Solution):
                _data.aux['H_uu'] = np.array(
                        [_compute_h_uu(ti, xi, lami, ui, _data.p, _data.k)
                         for ti, xi, lami, ui in zip(_data.t, _data.x.T, _data.lam.T, _data.u.T)])

                # _data.aux['H_xx'] = np.array(
                #         [_compute_h_xx(ti, xi, lami, ui, _data.p, _data.k)
                #          for ti, xi, lami, ui in zip(_data.t, _data.x.T, _data.lam.T, _data.u.T)])

                # _data.aux['H_ux'] = np.array(
                #         [_compute_h_ux(ti, xi, lami, ui, _data.p, _data.k)
                #          for ti, xi, lami, ui in zip(_data.t, _data.x.T, _data.lam.T, _data.u.T)])

                # _data.aux['H_ulam'] = np.array(
                #         [_compute_h_ulam(ti, xi, lami, ui, _data.p, _data.k)
                #          for ti, xi, lami, ui in zip(_data.t, _data.x.T, _data.lam.T, _data.u.T)])

                return _data

            self.post_processes.append(compute_huu)

        else:
            warnings.warn(f'\"{name}\" is not a name for a post-process.')
