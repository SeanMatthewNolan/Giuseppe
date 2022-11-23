import copy
from enum import Enum
from typing import Optional, Callable
import tkinter as tk
from tkinter import ttk

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.io import Solution
from giuseppe.problems.dual import CompDualOCP

from ..constant import initialize_guess_for_auto
from ..propagation.simple import propagate_guess
from ..projection import match_constants_to_bcs, project_dual
from giuseppe.visualization.components.tk_widgets.range_selector import TKRangeSelector
from giuseppe.visualization.components.tk_widgets.control_editor import TKControlEditor
from giuseppe.visualization.components.tk_widgets.sol_viewer import TKSolViewer, SolutionComponentType
from giuseppe.visualization.components.tk_widgets.static_value_editor import TKStaticValueEditor


class StatusLabels(Enum):
    YES = 'Consistent: ✅'
    NO = 'Consistent: ❌'


class InteractiveGuessGenerator(tk.Tk):
    def __init__(
            self,
            prob: CompDualOCP,
            init_guess: Optional[Solution] = None,
            num_data_viewers: int = 3,
            inter_func: str = 'pchip'
    ):
        super().__init__()

        self.prob = prob

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
            self._guess = initialize_guess_for_auto(prob)
        else:
            self._guess = copy.copy(init_guess)

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
        self.initial_costate_editor.frame.grid(row=1, column=0, padx=6, pady=6, sticky=tk.N)

        self.constant_editor = TKStaticValueEditor(self.static_panel, self._guess.k, label='Constants')
        self.constant_editor.frame.grid(row=2, column=0, padx=6, pady=6, sticky=tk.N)

        self.parameter_editor = TKStaticValueEditor(self.static_panel, self._guess.p, label='Parameters')
        if len(self._guess.p) > 0:
            self.parameter_editor.frame.grid(row=3, column=0, padx=6, pady=6, sticky=tk.N)

        self.control_panel = ttk.Frame(self)
        self.control_panel.grid(row=1, column=0, columnspan=3, sticky=tk.NSEW)

        self.status_label = ttk.Label(self.control_panel, text=StatusLabels.YES.value)
        self.status_label.grid(row=0, column=0, padx=6, pady=6)

        self.t_span_control = TKRangeSelector(
                self.control_panel, label='Independent Span', lower=self._guess.t[0], upper=self._guess.t[-1],
                bindings=[lambda _: self.update_t_range()]
        )
        self.t_span_control.frame.grid(row=0, column=1, padx=6, pady=6)
        self.update_t_range()

        self.propagate_button = ttk.Button(self.control_panel, text='Propagate', command=self.propagate_guess)
        self.propagate_button.grid(row=0, column=2, padx=6, pady=6)

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

        # Update data viewers
        for viewer in self.data_viewers:
            viewer.sol = self._guess
            viewer.update()

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

    def propagate_guess(self):
        self.guess = propagate_guess(
                self.prob,
                initial_states=self.initial_state_editor.values,
                initial_costates=self.initial_costate_editor.values,
                k=self.constant_editor.values,
                p=self.guess.p,
                control=self.generate_control_func(),
                t_span=self.t_span_control.get(),
                use_project_dual=False,
                use_match_constants=False
        )

    def match_constants(self):
        self.guess = match_constants_to_bcs(self.prob, self._guess)

    def run(self) -> Solution:
        self.mainloop()
        return self.guess
