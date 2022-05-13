import sys
from typing import Optional

import tqdm

from .base import ContinuationMonitor
from ..methods import ContinuationSeries
from ...utils.strings import justify_str

BAR_LEN = 40
DESC_LEN = 40


# TODO Investigate allowing logging
class ProgressBarMonitor(ContinuationMonitor):
    def __init__(self, smoothing: float = 0.9):
        super().__init__()
        self.progress_bar: Optional[tqdm.tqdm] = None

        self.bar_format: str = ''
        self.smoothing: float = smoothing

    def initialize_progress_bar(self, desc='Continuation Series', total=None):
        self.progress_bar = tqdm.tqdm(
                desc=justify_str(desc, DESC_LEN), total=total, unit='sols', file=sys.stdout, smoothing=self.smoothing,
                bar_format='{l_bar}{bar:' + str(BAR_LEN) + '}{r_bar}'
        )

    def start_cont_series(self, series: ContinuationSeries):
        super().start_cont_series(series)

        if hasattr(series, 'num_steps'):
            total = series.num_steps
        else:
            total = None

        if hasattr(series, 'form_mapping_str'):
            desc = series.form_mapping_str()
        else:
            desc = repr(series)

        self.initialize_progress_bar(desc=desc, total=total)

    def log_step(self):
        super().log_step()
        self.progress_bar.update()

    def end_cont_series(self):
        self.progress_bar.close()
        self.progress_bar = None

    def log_msg(self, msg: str):
        if self.progress_bar is None:
            super().log_msg(msg)
        else:
            self.messages.append(msg)
            self.progress_bar.write(msg)
