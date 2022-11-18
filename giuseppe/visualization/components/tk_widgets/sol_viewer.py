import enum
from typing import Optional
import tkinter as tk
from tkinter import ttk
from tkinter.constants import NSEW

import numpy as np

from giuseppe.io import Solution
from giuseppe.visualization.components.tk_widgets.data_viewer import TKDataViewer
from giuseppe.utils.typing import EMPTY_ARRAY


class SolutionComponentType(enum.Enum):
    INDEPENDENT = 'independent'
    STATES = 'states'
    COSTATES = 'costates'
    CONTROLS = 'controls'


class DataSelector:
    def __init__(
            self,
            master: tk.Tk,
            comp_type: SolutionComponentType = SolutionComponentType.INDEPENDENT,
            idx: int = 0,
            max_idx: int = 0,
            label:  Optional[str] = None
    ):
        self.comp_type = comp_type
        self.idx = idx
        self.max_idx = max_idx

        if label is None:
            self.frame = ttk.Frame(master)
        else:
            self.frame = ttk.LabelFrame(master, text=label)

        self.tk_comp_type = tk.StringVar()
        self.comp_type_mapping = {
            'Independent': SolutionComponentType.INDEPENDENT,
            'State': SolutionComponentType.STATES,
            'Control': SolutionComponentType.CONTROLS,
            'Costates': SolutionComponentType.COSTATES
        }
        self.tk_comp_type.set(list(self.comp_type_mapping.keys())[0])
        self.type_box = ttk.Combobox(self.frame, textvariable=self.tk_comp_type)
        self.type_box['values'] = list(self.comp_type_mapping.keys())
        self.type_box['state'] = 'readonly'
        # self.type_box.grid(row=0, column=0, sticky=NSEW)
        self.type_box.pack()
        self.type_box.bind('<<ComboboxSelected>>', self._type_selected)

        self.tk_idx = tk.IntVar()
        self.tk_idx.set(1)
        self.idx_spinbox = ttk.Spinbox(self.frame, from_=1, to=max_idx+1, increment=1, textvariable=self.tk_idx)
        self.idx_spinbox.pack(pady=5)

        self.pack = self.frame.pack
        self.grid = self.frame.grid

    def _type_selected(self, event: tk.Event):
        self.comp_type = self.comp_type_mapping[self.tk_comp_type.get()]
        print(self.tk_comp_type.get())
        print(self.comp_type)

    def _idx_selected(self, event: tk.Event):
        self.idx = self.comp_type_mapping[self.tk_comp_type.get()]
        print(self.tk_comp_type.get())
        print(self.comp_type)


class TKSolViewer(TKDataViewer):
    def __init__(
            self,
            master: tk.Tk,
            sol: Solution,
            hor_type: SolutionComponentType = SolutionComponentType.INDEPENDENT,
            hor_idx: int = 0,
            vert_type: SolutionComponentType = SolutionComponentType.STATES,
            vert_idx: int = 0,
    ):
        super().__init__(master)
        self.sol: Solution = sol
        self.types: tuple[SolutionComponentType, SolutionComponentType] = (hor_type, vert_type)
        self.indices: tuple[int, int] = (hor_idx, vert_idx)

        self.set_data(hor_type, hor_idx, vert_type, vert_idx)

        self.hor_data_selector = DataSelector(self.frame, label='X-Axis Data')
        self.hor_data_selector.pack(side=tk.LEFT, padx=3, pady=3)
        # self.hor_data_selector.grid(row=2, column=1, sticky=NSEW)

        self.vert_data_selector = DataSelector(self.frame, label='Y-Axis Data')
        self.vert_data_selector.pack(side=tk.RIGHT, padx=3, pady=3)
        # self.vert_data_selector.grid(row=1, column=0, sticky=NSEW)

    def set_data(
            self,
            h_type: SolutionComponentType = SolutionComponentType.INDEPENDENT,
            h_idx: int = 0,
            v_type: SolutionComponentType = SolutionComponentType.STATES,
            v_idx: int = 0
    ):
        h_data = self._get_data_slice(h_type, h_idx)
        v_data = self._get_data_slice(v_type, v_idx)

        if (h_data.shape != v_data.shape) or h_data.ndim != 1:
            h_data, v_data = EMPTY_ARRAY, EMPTY_ARRAY

        super().set_data(h_data, v_data)

    def _get_data_array(self, comp_type: SolutionComponentType) -> Optional[np.ndarray]:
        if comp_type == SolutionComponentType.INDEPENDENT:
            return self.sol.t
        elif comp_type == SolutionComponentType.STATES:
            return self.sol.x
        elif comp_type == SolutionComponentType.CONTROLS:
            return self.sol.u
        elif comp_type == SolutionComponentType.COSTATES:
            return self.sol.lam
        else:
            raise ValueError('Component type not found')

    def _get_num_elements(self, comp_type: SolutionComponentType) -> int:
        _data_array = self._get_data_array(comp_type)

        if _data_array is None:
            return 0
        elif _data_array.ndim == 1:
            return 1
        else:
            return _data_array.shape[0]

    def _get_data_slice(self, comp_type: SolutionComponentType, idx: int) -> np.ndarray:
        _data_array = self._get_data_array(comp_type)
        if _data_array is None:
            return EMPTY_ARRAY
        elif _data_array.ndim == 1:
            return _data_array
        else:
            if idx <= 0:
                return _data_array[0, :]
            elif idx >= _data_array.shape[0]:
                return _data_array[-1, :]
            else:
                return _data_array[idx, :]
