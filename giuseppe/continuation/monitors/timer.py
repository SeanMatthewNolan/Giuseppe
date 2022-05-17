from time import perf_counter_ns
from typing import Optional

from .base import ContinuationMonitor
from ...utils.timer import Timer, TIMER_TYPE, LOG_FUNC_TYPE


class ContinuationTimer(Timer, ContinuationMonitor):
    def __init__(self, prefix: str = '', timer_function: TIMER_TYPE = perf_counter_ns,
                 log_func: Optional[LOG_FUNC_TYPE] = print):
        Timer.__init__(self, prefix=prefix, timer_function=timer_function, log_func=log_func)
        ContinuationMonitor.__init__(self)

    def __enter__(self):
        Timer.__enter__(self)
        ContinuationMonitor.__enter__(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        Timer.__exit__(self, exc_type, exc_val, exc_tb)
        ContinuationMonitor.__exit__(self, exc_type, exc_val, exc_tb)

