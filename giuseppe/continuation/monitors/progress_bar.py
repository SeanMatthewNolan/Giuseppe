import sys
from typing import Optional

import tqdm

from .base import ContinuationMonitor
from ..methods import ContinuationSeries


# TODO Investigate allowing logging
class ProgressBarMonitor(ContinuationMonitor):
    def __init__(self, bar_len: int = 120, smoothing: float = 0.3):
        super().__init__()
        self.progress_bar: Optional[tqdm.tqdm] = None

        self.bar_len: int = bar_len
        self.smoothing: float = smoothing

    def initialize_progress_bar(self, desc='Continuation Series', total=None):
        self.progress_bar = tqdm.tqdm(
                desc=desc, total=total, unit='sols', file=sys.stdout, smoothing=self.smoothing, ncols=self.bar_len,
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
        # self.initialize_progress_bar(total=total)

    def log_step(self):
        super().log_step()
        # self.progress_bar.set_postfix({'a': 1}, refresh=False)
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
