import copy
from enum import Enum
from typing import Optional, Callable
import tkinter as tk
from tkinter import ttk

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import Dual
from giuseppe.numeric_solvers import NumericSolver

from giuseppe.visualization.components.tk_widgets.range_selector import TKRangeSelector
from giuseppe.visualization.components.tk_widgets.control_editor import TKControlEditor
from giuseppe.visualization.components.tk_widgets.sol_viewer import TKSolViewer, SolutionComponentType
from giuseppe.visualization.components.tk_widgets.static_value_editor import TKStaticValueEditor
from giuseppe.visualization.components.tk_widgets.static_value_viewer import TKStaticValueViewer

from giuseppe.utils.typing import EMPTY_ARRAY

from ..initialize_guess import initialize_guess
from ..propagate_guess import propagate_guess
from ..gauss_newton import match_states_to_boundary_conditions, match_adjoints,\
    match_constants_to_boundary_conditions, match_adjoints_to_boundary_conditions


class StatusLabels(Enum):
    CONSISTENT = 'Consistent: ✅'
    NOT_CONSISTENT = 'Not Consistent: ❌'
    CONVERGED = 'Converged: ✅'
    NOT_CONVERGED = 'Not Converged: ❌'


class InteractiveGuessGenerator(tk.Tk):
    def __init__(
            self,
            prob: Dual,  # TODO Add support for OCP
            init_guess: Optional[Solution] = None,
            num_data_viewers: int = 3,
            inter_func: str = 'pchip',
            num_solver: Optional[NumericSolver] = None
    ):
        super().__init__()

        self.prob = prob
        # self.bc_funcs = generate_separated_bc_funcs(self.prob)

        self.style = ttk.Style(self)
        # self.style.theme_use('alt')
        # self.style.theme_use('clam')
        self.style.theme_use('default')
        # self.style.theme_use('classic')
        self.title('Guess Generator')
        self.wm_title('Guess Generator')

        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)

        self._guess: Solution

        if init_guess is None:
            self._guess = initialize_guess(prob)
        else:
            self._guess = copy.copy(init_guess)

        self.reverse: bool = False

        self.initial_bc_res: np.ndarray = EMPTY_ARRAY
        self.terminal_bc_res: np.ndarray = EMPTY_ARRAY
        self.total_bc_res_norm: float = 0
        self.compute_bc_res()

        self.control_editors = []
        self.control_notebook = ttk.Notebook(self)
        self.control_notebook.grid(row=0, column=0, sticky=tk.NSEW, padx=12)
        for idx, control in enumerate(self._guess.u):
            _control_editor = TKControlEditor(self.control_notebook, inter_func=inter_func)
            self.control_editors.append(_control_editor)
            self.control_notebook.add(_control_editor.frame, text=f'Control {idx + 1}')

        self.data_viewers = []
        self.data_notebook = ttk.Notebook(self)
        self.data_notebook.grid(row=0, column=1, sticky=tk.NSEW, padx=12)
        for idx in range(num_data_viewers):
            _data_viewer = TKSolViewer(
                    self, self._guess,
                    vert_type=SolutionComponentType.STATES, vert_idx=min(idx, len(self._guess.x) - 1))
            self.data_viewers.append(_data_viewer)
            self.data_notebook.add(_data_viewer.frame, text=f'Data Viewer {idx + 1}')

        self.static_panel = ttk.Frame(self, relief=tk.RIDGE)
        self.static_panel.grid(row=0, column=2)

        self.initial_state_editor = TKStaticValueEditor(self.static_panel, self._guess.x[:, 0], label='Initial States')
        self.initial_state_editor.frame.grid(row=0, column=0, padx=6, pady=6, sticky=tk.N)

        self.initial_costate_editor = TKStaticValueEditor(
                self.static_panel, self._guess.lam[:, 0], label='Initial Costates')
        self.initial_costate_editor.frame.grid(column=0, padx=6, pady=6, sticky=tk.N)

        self.initial_adjoint_editor = TKStaticValueEditor(self.static_panel, self._guess.nu0, label='Initial Adjoints')
        self.initial_adjoint_editor.frame.grid(column=0, padx=6, pady=6, sticky=tk.N)

        self.terminal_adjoint_editor = TKStaticValueEditor(
                self.static_panel, self._guess.nuf, label='Terminal Adjoints')
        self.terminal_adjoint_editor.frame.grid(column=0, padx=6, pady=6, sticky=tk.N)

        self.constant_editor = TKStaticValueEditor(self.static_panel, self._guess.k, label='Constants')
        self.constant_editor.frame.grid(column=0, padx=6, pady=6, sticky=tk.N)

        self.parameter_editor = TKStaticValueEditor(self.static_panel, self._guess.p, label='Parameters')
        if len(self._guess.p) > 0:
            self.parameter_editor.frame.grid(column=0, padx=6, pady=6, sticky=tk.N)

        self.error_panel = ttk.Frame(self, relief=tk.RIDGE)
        self.error_panel.grid(row=0, column=3)

        self.initial_bc_viewer = TKStaticValueViewer(
                self.error_panel, self.initial_bc_res, label='Initial BC Residuals')
        self.initial_bc_viewer.frame.grid(row=0, column=0, sticky=tk.NSEW)

        self.terminal_bc_viewer = TKStaticValueViewer(
                self.error_panel, self.terminal_bc_res, label='Terminal BC Residuals')
        self.terminal_bc_viewer.frame.grid(row=1, column=0, sticky=tk.NSEW)

        self.total_bc_viewer = TKStaticValueViewer(
                self.error_panel, [self.total_bc_res_norm], label='BC Residual Norm')
        self.total_bc_viewer.frame.grid(row=2, column=0, sticky=tk.NSEW)

        self.control_panel = ttk.Frame(self)
        self.control_panel.grid(row=1, column=0, columnspan=3, sticky=tk.NSEW, padx=6, pady=6)

        self.status_label = ttk.Label(self.control_panel, text=StatusLabels.NOT_CONSISTENT.value)
        self.status_label.grid(row=0, column=0, padx=6, pady=6)

        self.t_span_control = TKRangeSelector(
                self.control_panel, label='Independent Range', lower=self._guess.t[0], upper=self._guess.t[-1],
                bindings=[lambda _: self.update_t_range()]
        )
        self.t_span_control.frame.grid(row=0, column=0, padx=6, pady=6)
        self.update_t_range()

        self.match_states_button = ttk.Button(
                self.control_panel, text='Match Initial States', command=self.match_states)
        self.match_states_button.grid(row=0, column=1, padx=6, pady=6)

        self.propagate_button = ttk.Button(self.control_panel, text='Propagate', command=self.propagate_guess)
        self.propagate_button.grid(row=0, column=2, padx=6, pady=6)

        self.project_dual_button = ttk.Button(
                self.control_panel, text='Project Dual', command=self.match_adjoints)
        self.project_dual_button.grid(row=0, column=3, padx=6, pady=6)

        self.match_constants_button = ttk.Button(
                self.control_panel, text='Match Constants', command=self.match_constants)
        self.match_constants_button.grid(row=0, column=4, padx=6, pady=6)

        self.auto_button = ttk.Button(self.control_panel, text='Auto', command=self.auto)
        self.auto_button.grid(row=0, column=5, padx=6, pady=6)

        self.num_solver = num_solver
        if self.num_solver is not None:
            self.solve_button = ttk.Button(self.control_panel, text='Solve', command=self.solve)
            self.solve_button.grid(row=0, column=6, padx=6, pady=6)

            self.solve_status_label = ttk.Label(self.control_panel, text=StatusLabels.NOT_CONVERGED.value)
            self.solve_status_label.grid(row=0, column=7, padx=6, pady=6)

        # self.save_button = ttk.Button(self.control_panel, text='Save', command=self.save)
        # self.save_button.grid(row=0, column=7, padx=6, pady=6)

        # self.bind('<Return>', lambda _: self.propagate_guess(), add='+')

    @property
    def guess(self):
        return self._guess

    @guess.setter
    def guess(self, _guess):
        self._guess = _guess

        # Update t span
        self.t_span_control.lower, self.t_span_control.upper = self._guess.t[0], self._guess.t[-1]
        self.update_t_range()

        # Update statics
        self.parameter_editor.values = self._guess.p
        self.constant_editor.values = self._guess.k
        self.initial_adjoint_editor.values = self._guess.nu0
        self.terminal_adjoint_editor.values = self._guess.nuf
        self.initial_state_editor.values = self._guess.x[:, 0]
        self.initial_costate_editor.values = self._guess.lam[:, 0]

        # Update error
        self.compute_bc_res()
        self.initial_bc_viewer.values = self.initial_bc_res
        self.terminal_bc_viewer.values = self.terminal_bc_res
        self.total_bc_viewer.values = self.total_bc_res_norm

        # Update data viewers
        for viewer in self.data_viewers:
            viewer.sol = self._guess
            viewer.update()

    def compute_bc_res(self):
        _bc_res = self.prob.compute_boundary_conditions(self._guess.t, self._guess.x, self._guess.p, self._guess.k)
        _adj_bc_res = self.prob.compute_adjoint_boundary_conditions(
                self._guess.t, self._guess.x, self._guess.lam, self._guess.u,
                self._guess.p, np.concatenate((self._guess.nu0, self._guess.nuf)), self._guess.k)

        self.initial_bc_res = _bc_res
        self.terminal_bc_res = _bc_res
        self.total_bc_res_norm = np.linalg.norm(np.concatenate((self.initial_bc_res, self.terminal_bc_res)))

    def generate_control_func(self) -> Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike]:

        if len(self.control_editors) == 1:
            inter = self.control_editors[0].inter

            def control_func(t, _, __, ___):
                return np.array([inter(t)])

        else:
            splines = [control_editor.inter for control_editor in self.control_editors]

            def control_func(t, _, __, ___):
                return np.array([_inter(t) for _inter in splines])

        return control_func

    def update_t_range(self):
        for control_editor in self.control_editors:
            control_editor.set_t_range(self.t_span_control.lower, self.t_span_control.upper)

    def match_states(self):
        if self.reverse:
            loc_idx = -1
        else:
            loc_idx = 0

        self.initial_state_editor.values = \
            match_states_to_boundary_conditions(self.prob, self._guess).x[:, loc_idx]
        self.initial_costate_editor.values = \
            match_adjoints_to_boundary_conditions(self.prob, self._guess).lam[:, loc_idx]

    def propagate_guess(self):
        self.guess = propagate_guess(
                self.prob,
                initial_states=self.initial_state_editor.values,
                initial_costates=self.initial_costate_editor.values,
                k=self.constant_editor.values,
                p=self.guess.p,
                nu0=self.initial_adjoint_editor.values,
                nuf=self.terminal_adjoint_editor.values,
                control=self.generate_control_func(),
                t_span=self.t_span_control.get(),
                reverse=self.reverse
        )

    def match_adjoints(self):
        self.guess = match_adjoints(self.prob, self._guess)

    def match_constants(self):
        self.guess = match_constants_to_boundary_conditions(self.prob, self._guess)

    def auto(self):
        self.match_states()
        self.propagate_guess()
        self.match_adjoints()
        self.match_constants()

    def solve(self):
        if self.num_solver is not None:
            _sol = self.num_solver.solve(self._guess)
            if _sol.converged:
                self.guess = _sol
                self.solve_status_label['text'] = StatusLabels.CONVERGED.value
            else:
                self.solve_status_label['text'] = StatusLabels.NOT_CONVERGED.value
            self.num_solver.verbose = False

    def save(self):
        pass

    def run(self) -> Solution:
        self.mainloop()
        return self.guess
