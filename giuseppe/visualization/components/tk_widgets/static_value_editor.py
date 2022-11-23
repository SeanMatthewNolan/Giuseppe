from typing import Optional
import tkinter as tk
from tkinter import ttk

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.io import Solution


class TKStaticValueEditor:
    def __init__(
            self,
            master,
            constants: ArrayLike = np.array([]),
            label: Optional[str] = None
    ):
        self._master = master
        self._label = label

        self._values = np.asarray(constants)
        self._form_entry_fields()

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, _values: ArrayLike):
        self._values = np.asarray(_values)
        if len(self._values) != len(_values):
            self._form_entry_fields()
        self.update()

    def update(self):
        for idx, tk_var in enumerate(self.tk_constant_vars):
            tk_var.set(f'{self._values[idx]:g}')

    def _form_entry_fields(self):
        if self._label is None:
            self.frame = ttk.Frame(self._master)
        else:
            self.frame = ttk.LabelFrame(self._master, text=self._label)

        self.constant_labels = []
        self.tk_constant_vars = []
        self.constant_editors = []
        for idx, constant in enumerate(self._values):
            _label = ttk.Label(self.frame, text=f'Value {idx + 1}')
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
                self._values[idx] = float(tk_var.get())
                tk_var.set(f'{self._values[idx]:g}')
            except ValueError:
                tk_var.set(f'{self._values[idx]:g}')

        return _constant_binding
