from typing import Optional, Iterable
import tkinter as tk
from tkinter import ttk

import numpy as np
from numpy.typing import ArrayLike


class TKStaticValueViewer:
    def __init__(
            self,
            master,
            values: ArrayLike = np.array([]),
            label: Optional[str] = None
    ):
        self._master = master
        self._label = label

        self._values = np.asarray(values)
        self._form_value_fields()

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, _values: ArrayLike):
        if not isinstance(_values, Iterable):
            _values = [_values]

        self._values = np.asarray(_values)
        if len(self._values) != len(_values):
            self._form_value_fields()
        self.update()

    def update(self):
        for idx, view in enumerate(self.value_views):
            view['text'] = f'{self._values[idx]:6.4f}'

    def _form_value_fields(self):
        if self._label is None:
            self.frame = ttk.Frame(self._master)
        else:
            self.frame = ttk.LabelFrame(self._master, text=self._label)

        self.constant_labels = []
        self.value_views = []
        for idx, val in enumerate(self._values):
            # TODO Add labels based off annotations
            # _label = ttk.Label(self.frame, text=f'Value {idx + 1}')
            # self.constant_labels.append(_label)
            # _label.grid(row=idx, column=0)

            _value_view = ttk.Label(self.frame, text=f'{val:6.4f}')
            self.value_views.append(_value_view)
            _value_view.grid(row=idx, column=0)
