from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .range_selector import TKRangeSelector
from ..matplotlib_elements import SplineEditor


class TKControlEditor(SplineEditor):
    def __init__(self,
                 master: tk.Widget,
                 t_range: tuple[float, float] = (0., 1.),
                 u_range: tuple[float, float] = (-1., 1.),
                 inter_func: str = 'pchip',
                 ):

        self.frame = ttk.Frame(master, padding='3 3 12 12', relief=tk.RIDGE)
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

        self.range_selector = TKRangeSelector(
                self.control_panel, lower=self.y_range[0], upper=self.y_range[1], label='Control Range',
                bindings=self._set_u_range_from_selector)
        # self.range_selector.pack(side=tk.LEFT)
        self.range_selector.grid(row=1, column=0, padx=6, pady=6)

        self.inter_options = ['PCHIP', 'Linear', 'Spline', 'Akima', 'Krogh', 'Quadratic', 'Nearest', 'Previous', 'Next']
        self.tk_inter_options = tk.StringVar()
        self.tk_inter_options.set(self.inter_options[0])
        self.inter_box = ttk.Combobox(self.control_panel, textvariable=self.tk_inter_options)
        self.inter_box['values'] = self.inter_options
        self.inter_box['state'] = 'readonly'
        self.inter_box.bind('<<ComboboxSelected>>', self._combox_set_inter, add='+')
        self.inter_box.grid(row=1, column=1, padx=6, pady=6)

        self.reset_button = ttk.Button(self.control_panel, text='Reset', command=self.reset)
        self.reset_button.grid(row=1, column=2, padx=6, pady=6)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.control_panel, pack_toolbar=False)
        self.toolbar.update()
        # self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar.grid(row=0, column=0, columnspan=3, sticky=tk.NSEW)

        self.control_panel.pack(fill=tk.X)

    def _set_u_range_from_selector(self, _):
        self.set_u_range(lower=self.range_selector.lower, upper=self.range_selector.upper)
        self.nodes[1, :] = np.clip(self.nodes[1, :], min(self.y_range), max(self.y_range))

    def _combox_set_inter(self, _):
        self.set_interpolator(inter_func=self.inter_box.get())

