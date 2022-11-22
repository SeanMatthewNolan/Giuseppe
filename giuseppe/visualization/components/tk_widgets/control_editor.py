import tkinter as tk
from tkinter import ttk, RIDGE
from tkinter.constants import NSEW, EW, LEFT
from typing import Optional, Callable, Union, Iterable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ..matplotlib_elements import SplineEditor


class RangeSector:
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
            self.frame.bind('<FocusOut>', binding, add='+')
            self.upper_entry.bind('<Return>', binding, add='+')
            self.lower_entry.bind('<Return>', binding, add='+')


class TKControlEditor(SplineEditor):
    def __init__(self,
                 master: tk.Tk,
                 t_range: tuple[float, float] = (0., 1.),
                 u_range: tuple[float, float] = (-1., 1.),
                 inter_func: str = 'pchip',
                 ):

        self.frame = ttk.Frame(master, padding='3 3 12 12', relief=RIDGE)
        # self.frame.columnconfigure(0, weight=1)
        # self.frame.rowconfigure(0, weight=1)

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot()
        fig.canvas = FigureCanvasTkAgg(fig, master=self.frame)

        super().__init__(axes=ax, x_range=t_range, y_range=u_range, inter_func=inter_func)
        self.set_t_range = self.set_x_range
        self.set_u_range = self.set_y_range

        self.fig.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # self.fig.canvas.get_tk_widget().grid(row=0, column=1, sticky=NSEW, columnspan=True)

        self.control_panel = ttk.Frame(self.frame)

        self.range_selector = RangeSector(
                self.control_panel, lower=self.y_range[0], upper=self.y_range[1], label='Control Range',
                bindings=self._set_u_range_from_selector)
        # self.range_selector.pack(side=tk.LEFT)
        self.range_selector.grid(row=0, column=0, padx=5)

        self.inter_options = ['PCHIP', 'Linear', 'Spline', 'Akima', 'Krogh', 'Quadratic', 'Nearest', 'Previous', 'Next']
        self.tk_inter_options = tk.StringVar()
        self.tk_inter_options.set(self.inter_options[0])
        self.inter_box = ttk.Combobox(self.control_panel, textvariable=self.tk_inter_options)
        self.inter_box['values'] = self.inter_options
        self.inter_box['state'] = 'readonly'
        self.inter_box.bind('<<ComboboxSelected>>', self._combox_set_inter, add='+')
        self.inter_box.grid(row=0, column=1, padx=5)

        self.reset_button = ttk.Button(self.control_panel, text='Reset', command=self.reset)
        self.reset_button.grid(row=0, column=2)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.control_panel, pack_toolbar=False)
        self.toolbar.update()
        # self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar.grid(row=1, column=0, columnspan=3, sticky=NSEW)

        self.control_panel.pack()

    def _set_u_range_from_selector(self, _):
        self.set_u_range(lower=self.range_selector.lower, upper=self.range_selector.upper)
        self.nodes[1, :] = np.clip(self.nodes[1, :], min(self.y_range), max(self.y_range))

    def _combox_set_inter(self, _):
        self.set_interpolator(inter_func=self.inter_box.get())

