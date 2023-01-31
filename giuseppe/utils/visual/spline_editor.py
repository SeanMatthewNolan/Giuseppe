from __future__ import annotations

from typing import Optional, Callable, Union

import matplotlib.backend_bases
import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backend_tools import Cursors

InterpolatorType = Callable[[ArrayLike, ArrayLike], Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]]


def form_interpolator(inter_func: str = 'pchip') -> InterpolatorType:
    if inter_func.lower() == 'pchip':
        interpolator = interpolate.PchipInterpolator
    elif inter_func.lower() == 'linear':
        interpolator = interpolate.interp1d
    elif inter_func.lower() == 'spline':
        interpolator = interpolate.CubicSpline
    elif inter_func.lower() == 'akima':
        interpolator = interpolate.Akima1DInterpolator
    elif inter_func.lower() == 'krogh':
        interpolator = interpolate.KroghInterpolator
    elif inter_func.lower() == 'nearest':
        def _interpolator(_x, _y):
            return interpolate.interp1d(_x, _y, kind='nearest')
        interpolator = _interpolator
    elif inter_func.lower() == 'previous':
        def _interpolator(_x, _y):
            return interpolate.interp1d(_x, _y, kind='previous')
        interpolator = _interpolator
    elif inter_func.lower() == 'next':
        def _interpolator(_x, _y):
            return interpolate.interp1d(_x, _y, kind='next')
        interpolator = _interpolator
    elif inter_func.lower() == 'quadratic':
        def _interpolator(_x, _y):
            return interpolate.interp1d(_x, _y, kind='quadratic')
        interpolator = _interpolator
    else:
        raise RuntimeError(f'Interpolation function {inter_func} not available')

    return interpolator


class SplineEditor:
    def __init__(
            self,
            axes: Optional[plt.Axes] = None,
            x_range: tuple[float, float] = (0., 1.),
            num_points: int = 3,
            y_range: tuple[float, float] = (-1., 1),
            inter_func: str = 'pchip'
    ):
        self.interpolator: InterpolatorType = form_interpolator(inter_func)

        if axes is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax: plt.Axes = axes
            self.fig: plt.Figure = self.ax.figure

        self.canvas = self.fig.canvas

        self.x_range: tuple[float, float] = x_range
        self.y_range: tuple[float, float] = y_range
        self.set_ax_lims()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.num_points: int = num_points
        self.nodes: np.ndarray = np.vstack((
            np.linspace(self.x_range[0], self.x_range[1], self.num_points),
            sum(y_range) / 2 * np.ones(self.num_points)
        ))

        self._node_line = Line2D(
                self.nodes[0, :], self.nodes[1, :],
                marker='o', linestyle='', markeredgecolor='C1', markerfacecolor='w', animated=True,
        )
        self.ax.add_line(self._node_line)
        self._node_idx: Optional[int] = None

        self.inter_res = 100
        self._inter_x = np.linspace(self.x_range[0], self.x_range[1], self.inter_res)
        self.inter = self.interpolator(self.nodes[0, :], self.nodes[1, :])
        self._inter_line = Line2D(
                self._inter_x, self.inter(self._inter_x), animated=True,
        )
        self.ax.add_line(self._inter_line)

        self.click_margin: int = 10
        self.node_margin: float = 0.01 * (self.x_range[1] - self.x_range[0]) / self.num_points

        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.canvas.draw()

    def set_ax_lims(self):
        # noinspection PyTypeChecker
        self.ax.set_xlim(self.x_range)
        # noinspection PyTypeChecker
        self.ax.set_ylim(self.y_range)

    def update_nodes(self):
        self._node_line.set_xdata(self.nodes[0, :])
        self._node_line.set_ydata(self.nodes[1, :])

        self.ax.draw_artist(self._node_line)

    def set_node(self, idx, x, y):
        if idx == 0:
            # x_min, x_max = self.x_span[0], self.nodes[0, 1] - self.node_margin
            self.nodes[1, 0] = np.clip(y, self.y_range[0], self.y_range[1])
        elif idx == self.num_points - 1:
            # x_min, x_max = self.nodes[0, idx - 1] + self.node_margin, self.x_span[1]
            self.nodes[1, -1] = np.clip(y, self.y_range[0], self.y_range[1])
        else:
            x_min, x_max = self.nodes[0, idx - 1] + self.node_margin, self.nodes[0, idx + 1] - self.node_margin
            self.nodes[:, self._node_idx] = np.clip(x, x_min, x_max), np.clip(y, self.y_range[0], self.y_range[1])

    def update_inter(self):
        self.inter = self.interpolator(self.nodes[0, :], self.nodes[1, :])
        _inter_x_without_nodes = np.linspace(self.x_range[0], self.x_range[1], self.inter_res)
        self._inter_x = \
            np.insert(_inter_x_without_nodes, _inter_x_without_nodes.searchsorted(self.nodes[0, :]), self.nodes[0, :])
        # noinspection PyTypeChecker
        self._inter_line.set_xdata(self._inter_x)
        # noinspection PyTypeChecker
        self._inter_line.set_ydata(self.inter(self._inter_x))

        self.ax.draw_artist(self._inter_line)

    def quick_draw(self):
        self.canvas.restore_region(self.background)
        # self.set_ax_lims()
        self.update_nodes()
        self.update_inter()
        self.canvas.blit(self.ax.bbox)

    def get_node_under_point(self, event):
        node_pixels = self._node_line.get_transform().transform(self.nodes.T).T
        dist = np.hypot(node_pixels[0, :] - event.x, node_pixels[1, :] - event.y)
        node_idx = np.argmin(dist)

        if dist[node_idx] > self.click_margin:
            node_idx = None

        return node_idx

    def on_draw(self, _):
        # self.set_ax_lims()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.update_nodes()
        self.update_inter()

    def on_button_press(self, event):
        # button pressed None, 1, 2, 3, ‘up’, ‘down’ (up and down are used for scroll events)
        if event.button == 1:
            self._node_idx = self.get_node_under_point(event)

    def on_button_release(self, event):
        if event.button == 1:
            self._node_idx = None
            if not self.canvas.widgetlock.locked():
                self.canvas.set_cursor(Cursors.POINTER)

    def on_mouse_move(self, event):
        if self._node_idx is None:
            return None
        elif event.inaxes is None:
            return None
        elif event.button == 1:
            self.canvas.set_cursor(Cursors.SELECT_REGION)
            self.set_node(self._node_idx, event.xdata, event.ydata)
            self.quick_draw()
        else:
            return None

    def change_interpolator(self, inter_func: str = 'pchip'):
        self.interpolator = form_interpolator(inter_func)
