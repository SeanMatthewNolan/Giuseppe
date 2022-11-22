import copy
from typing import Optional
import tkinter as tk
from tkinter import ttk

from giuseppe.io import Solution
from giuseppe.problems.dual import CompDualOCP

from ..constant import initialize_guess_for_auto
from giuseppe.visualization.components.tk_widgets.control_editor import TKControlEditor
from giuseppe.visualization.components.tk_widgets.sol_viewer import TKSolViewer, SolutionComponentType
from giuseppe.visualization.components.tk_widgets.constant_editor import TKConstantEditor


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

        if init_guess is None:
            self.guess: Solution = initialize_guess_for_auto(prob)
        else:
            self.guess: Solution = copy.copy(init_guess)

        self.control_editors = []
        self.control_notebook = ttk.Notebook(self)
        self.control_notebook.grid(row=0, column=0, sticky=tk.NSEW, padx=12)
        for idx, control in enumerate(self.guess.u):
            _control_editor = TKControlEditor(self.control_notebook, inter_func=inter_func)
            self.control_editors.append(_control_editor)
            self.control_notebook.add(_control_editor.frame, text=f'Control {idx + 1}')

        self.data_viewers = []
        self.data_notebook = ttk.Notebook(self)
        self.data_notebook.grid(row=0, column=1, sticky=tk.NSEW, padx=12)
        for idx in range(num_data_viewers):
            _data_viewer = TKSolViewer(
                    self, self.guess,
                    vert_type=SolutionComponentType.STATES, vert_idx=min(idx, len(self.guess.x) - 1))
            self.data_viewers.append(_data_viewer)
            self.data_notebook.add(_data_viewer.frame, text=f'Data Viewer {idx + 1}')

        self.constant_editor = TKConstantEditor(self, self.guess, label='Constants')
        self.constant_editor.frame.grid(row=0, column=2, sticky=tk.N)

        self.bind('<Return>', self.propagate, add='+')

    def propagate(self, event):
        print('PROPAGATE')

    def run(self) -> Solution:
        self.mainloop()
        return self.guess
