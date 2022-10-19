from typing import Optional

import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc

from ..io import Solution, SolutionSet


class DashSolutionViewer:
    def __init__(self, solution: Solution):
        self.sol: Solution = solution
        self.app: Optional[Dash] = None

    @staticmethod
    def update_figure(_x: np.ndarray, _y: np.ndarray) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=_x, y=_y, mode='lines+markers'))
        return fig

    def start_app(self):
        self.app = Dash('Solution Viewer')

        self.app.layout = html.Div(
            children=[
                html.H1(children='Hello'),
                dcc.Graph(
                    id='solution-view',
                    figure=self.update_figure()
                )
            ]
        )

        self.app.run_server()
