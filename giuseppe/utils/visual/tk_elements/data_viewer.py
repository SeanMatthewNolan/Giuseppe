import tkinter as tk
from tkinter import ttk, RIDGE

import numpy as np
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class TKDataViewer:
    def __init__(self, master=None):

        self.frame = ttk.Frame(master, padding='3 3 12 12', relief=RIDGE)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot()

        self._data_line = Line2D([], [])
        self.ax.add_line(self._data_line)

        self.fig.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas = self.fig.canvas
        self.fig.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def set_data(self, x: np.ndarray, y: np.ndarray):
        self._data_line.set_data(x, y)
