import sys
from typing import Optional

import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .base import ContinuationMonitor
from ..methods import ContinuationSeries
from ...utils.strings import justify_str

BAR_LEN = 40
DESC_LEN = 40


class ProgressBarMonitor(ContinuationMonitor):
    def __init__(self):
        super().__init__()
        self.progress_bar: Optional[tqdm.tqdm] = None

        self.bar_format: str = '{l_bar}{bar:' + str(BAR_LEN) + '}{r_bar}'
        self.smoothing: float = 0.9
        # tqdm default is 0.3 but nature of cont. makes more recent iter. more representative

        self._log_redirect = None
        self._orig_out_err = None

    def __enter__(self):
        # Redirect logging
        self._log_redirect = logging_redirect_tqdm()
        # noinspection PyUnresolvedReferences
        self._log_redirect.__enter__()

        # Redirect sys.stdout and sys.stderr
        self._orig_out_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = map(tqdm.contrib.DummyTqdmFile, self._orig_out_err)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar is not None:
            self.progress_bar.close()

        if self._orig_out_err is not None:
            sys.stdout, sys.stderr = self._orig_out_err

        if self._log_redirect is not None:
            self._log_redirect.__exit__(exc_type, exc_val, exc_tb)
            self._log_redirect = None

    def initialize_progress_bar(self, desc='Continuation Series', total=None):
        if self._orig_out_err is not None:
            tqdm_file = self._orig_out_err[0]
        else:
            tqdm_file = sys.stdout

        self.progress_bar = tqdm.tqdm(
                desc=justify_str(desc, DESC_LEN), total=total, unit='sols', file=tqdm_file, smoothing=self.smoothing,
                bar_format=self.bar_format, dynamic_ncols=True
        )

    def close_progress_bar(self):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

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
        self.close_progress_bar()

    def log_msg(self, msg: str):
        if self.progress_bar is None:
            super().log_msg(msg)
        else:
            self.messages.append(msg)
            self.progress_bar.write(msg)
