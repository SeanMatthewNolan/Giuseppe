from typing import Optional
import tkinter as tk
from tkinter import ttk
from tkinter.constants import NSEW

from giuseppe.io import Solution
from giuseppe.problems.dual import CompDualOCP

from ..constant import initialize_guess_for_auto
from giuseppe.visualization.components.tk_widgets.control_editor import TKControlEditor
from giuseppe.visualization.components.tk_widgets.sol_viewer import TKSolViewer


class InteractiveGuessGenerator:
    def __init__(self, prob: CompDualOCP, init_guess: Optional[Solution] = None, inter_func: str = 'pchip'):
        self.prob = prob

        self._padx = 12
        self._pady = 12

        self.root = tk.Tk()
        self.style = ttk.Style(self.root)
        # self.style.theme_use('alt')
        self.style.theme_use('clam')
        # self.style.theme_use('default')
        # self.style.theme_use('classic')
        self.root.title('Guess Generator')
        self.root.wm_title('Guess Generator')

        # self.mainframe = ttk.Frame(self.root, padding='3 3 12 12', relief='ridge')
        # self.mainframe.grid(column=0, row=0, sticky='nwes', padx=self._padx, pady=self._pady)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        if init_guess is None:
            self.guess: Solution = initialize_guess_for_auto(prob)
        else:
            self.guess: Solution = init_guess

        self.control_editors = []
        # for control in range(prob.comp_ocp.num_controls):
        for control in range(2):
            self.control_editors.append(TKControlEditor(self.root, inter_func=inter_func))
            self.control_editors[-1].frame.grid(row=0, column=control, sticky=NSEW, padx=self._padx, pady=self._pady)

        self.data_viewers = []
        for state in range(2):
            self.data_viewers.append(TKSolViewer(self.root, self.guess))
            self.data_viewers[-1].frame.grid(row=1, column=state, sticky=NSEW, padx=self._padx, pady=self._pady)

        self.root.bind('<Return>', self.propagate, add='+')

    def propagate(self, event):
        print('PROPAGATE')

    def run(self):
        self.root.mainloop()
