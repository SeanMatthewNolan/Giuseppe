from __future__ import annotations
from .base import ContinuationDisplayManager


class MultiDisplayManager(ContinuationDisplayManager):
    def __init__(self, monitors: list[ContinuationDisplayManager]):
        ContinuationDisplayManager.__init__(self)
        self.monitors: list[ContinuationDisplayManager] = monitors

    def __enter__(self):
        for monitor in self.monitors:
            monitor.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for monitor in self.monitors:
            monitor.__exit__(exc_type, exc_val, exc_tb)

    def start_cont_series(self, series):
        for monitor in self.monitors:
            monitor.start_cont_series(series)

    def log_step(self):
        for monitor in self.monitors:
            monitor.log_step()

    def log_msg(self, msg: str):
        for monitor in self.monitors:
            monitor.log_msg(msg)

    def end_cont_series(self):
        for monitor in self.monitors:
            monitor.end_cont_series()
