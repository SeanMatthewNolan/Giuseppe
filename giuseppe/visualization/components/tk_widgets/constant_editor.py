from typing import Optional
import tkinter as tk
from tkinter import ttk

from giuseppe.io import Solution


class TKConstantEditor:
    def __init__(
            self,
            master,
            sol: Solution,
            label: Optional[str] = None
    ):
        self.sol: Solution = sol

        if label is None:
            self.frame = ttk.Frame(master)
        else:
            self.frame = ttk.LabelFrame(master, text=label)

        self.constant_labels = []
        self.tk_constant_vars = []
        self.constant_editors = []
        for idx, constant in enumerate(self.sol.k):
            _label = ttk.Label(self.frame, text=f'Constant {idx + 1}')
            self.constant_labels.append(_label)
            _label.grid(row=idx, column=0)

            _tk_var = tk.StringVar()
            _tk_var.set(str(constant))
            self.tk_constant_vars.append(_tk_var)
            _editor = ttk.Entry(self.frame, textvariable=_tk_var)
            self.constant_editors.append(_editor)
            _func = self._generate_constant_binding(idx, _tk_var)
            _editor.bind('<FocusOut>', _func, add='+')
            _editor.bind('<Return>', _func, add='+')
            _editor.grid(row=idx, column=1)

    def _generate_constant_binding(self, idx, tk_var):
        def _constant_binding(_):
            try:
                self.sol.k[idx] = float(tk_var.get())
                tk_var.set(f'{self.sol.k[idx]:g}')
            except ValueError:
                tk_var.set(f'{self.sol.k[idx]:g}')

        return _constant_binding
