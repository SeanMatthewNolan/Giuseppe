import tkinter as tk
from tkinter import ttk, RIDGE

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class TKDataViewer:
    def __init__(self, master: tk.Tk, include_navbar: bool = True):

        self.frame = ttk.Frame(master, padding='3 3 12 12', relief=RIDGE)
        # self.frame.columnconfigure(1, weight=1)
        # self.frame.rowconfigure(1, weight=1)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax: Axes = self.fig.add_subplot()
        self.ax.set_autoscale_on(True)

        self._data_line = Line2D([], [])
        self.ax.add_line(self._data_line)

        self.fig.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas = self.fig.canvas
        self.fig.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # self.fig.canvas.get_tk_widget().grid(row=1, column=1, )

        if include_navbar:
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            # self.toolbar.grid(row=0, column=1)

    def set_data(self, x: np.ndarray, y: np.ndarray):
        # noinspection PyTypeChecker
        self._data_line.set_xdata(x)
        # noinspection PyTypeChecker
        self._data_line.set_ydata(y)
        self.ax.relim()
        self.ax.autoscale(tight=True)
        self.canvas.draw()
