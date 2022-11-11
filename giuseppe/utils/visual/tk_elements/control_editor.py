import tkinter as tk
from tkinter import ttk, RIDGE

from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from giuseppe.utils.visual import SplineEditor


class TKControlEditor(SplineEditor):
    def __init__(self,
                 t_range: tuple[float, float] = (0., 1.),
                 u_range: tuple[float, float] = (-1., 1.),
                 inter_func: str = 'pchip',
                 master=None
                 ):

        self.frame = ttk.Frame(master, padding='3 3 12 12', relief=RIDGE)
        # self.frame.columnconfigure(0, weight=1)
        # self.frame.rowconfigure(0, weight=1)

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot()
        fig.canvas = FigureCanvasTkAgg(fig, master=self.frame)

        super().__init__(axes=ax, x_range=t_range, y_range=u_range, inter_func=inter_func)

        self.fig.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
