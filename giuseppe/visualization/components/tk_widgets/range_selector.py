import tkinter as tk
from tkinter import ttk
from typing import Optional, Union, Callable, Iterable


class TKRangeSelector:
    def __init__(self, master, lower: float = -1., upper: float = 1.,
                 bindings: Optional[Union[Callable, Iterable[Callable]]] = None, label: Optional[str] = None):

        self.upper: float = upper
        self.lower: float = lower

        if label is None:
            self.frame = ttk.Frame(master)
        else:
            self.frame = ttk.LabelFrame(master, text=label)

        self.tk_lower = tk.StringVar()
        self.tk_lower.set(str(self.lower))
        self.lower_entry = ttk.Entry(self.frame, textvariable=self.tk_lower)
        self.lower_entry.bind('<FocusOut>', self.set_lower, add='+')
        self.lower_entry.bind('<Return>', self.set_lower, add='+')
        self.lower_entry.pack(side=tk.LEFT)

        self.tk_upper = tk.StringVar()
        self.tk_upper.set(str(self.upper))
        self.upper_entry = ttk.Entry(self.frame, textvariable=self.tk_upper)
        self.upper_entry.bind('<FocusOut>', self.set_upper, add='+')
        self.upper_entry.bind('<Return>', self.set_upper, add='+')
        self.upper_entry.pack(side=tk.RIGHT)

        if bindings is None:
            bindings = []
        elif not isinstance(bindings, Iterable):
            bindings = [bindings]
        self.bindings = list(bindings)
        self._set_bindings()

        self.pack = self.frame.pack
        self.grid = self.frame.grid

    def set_upper(self, _):
        try:
            self.upper = float(self.tk_upper.get())
            self.tk_upper.set(f'{self.upper:g}')
        except ValueError:
            self.tk_upper.set(f'{self.upper:g}')

    def set_lower(self, _):
        try:
            self.lower = float(self.tk_lower.get())
            self.tk_lower.set(f'{self.lower:g}')
        except ValueError:
            self.tk_lower.set(f'{self.lower:g}')

    def _set_bindings(self):
        for binding in self.bindings:
            self.upper_entry.bind('<FocusOut>', binding, add='+')
            self.lower_entry.bind('<FocusOut>', binding, add='+')
            self.upper_entry.bind('<Return>', binding, add='+')
            self.lower_entry.bind('<Return>', binding, add='+')

    def get(self):
        return self.lower, self.upper
