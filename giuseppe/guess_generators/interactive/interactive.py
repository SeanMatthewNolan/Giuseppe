import tkinter as tk
from tkinter.constants import NSEW

from giuseppe.problems.dual import CompDualOCP

from ..constant import initialize_guess_for_auto
from ...utils.visual.tk_elements.control_editor import TKControlEditor
from ...utils.visual.tk_elements.sol_viewer import TKSolViewer


class InteractiveGuessGenerator:
    def __init__(self, prob: CompDualOCP, inter_func: str = 'pchip'):
        self.prob = prob

        self._padx = 12
        self._pady = 12

        self.root = tk.Tk()
        self.root.title('Guess Generator')
        self.root.wm_title('Guess Generator')

        # self.mainframe = ttk.Frame(self.root, padding='3 3 12 12', relief='ridge')
        # self.mainframe.grid(column=0, row=0, sticky='nwes', padx=self._padx, pady=self._pady)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.guess = initialize_guess_for_auto(prob)

        self.control_editors = []
        # for control in range(prob.comp_ocp.num_controls):
        for control in range(2):
            self.control_editors.append(TKControlEditor(master=self.root, inter_func=inter_func))
            self.control_editors[-1].frame.grid(row=0, column=control, sticky=NSEW, padx=self._padx, pady=self._pady)

        self.data_viewers = []
        for state in range(2):
            self.data_viewers.append(TKSolViewer(self.guess, master=self.root))
            self.data_viewers[-1].frame.grid(row=1, column=state, sticky=NSEW, padx=self._padx, pady=self._pady)

        self.root.bind('<Return>', self.propagate)

    def propagate(self):
        pass

    def run(self):
        self.root.mainloop()
