from time import perf_counter_ns
from typing import Optional

from .base import ContinuationDisplayManager
from ...utils.timer import Timer, TIMER_TYPE, LOG_FUNC_TYPE


class ContinuationTimer(Timer, ContinuationDisplayManager):
    def __init__(self, prefix: str = '', timer_function: TIMER_TYPE = perf_counter_ns,
                 log_func: Optional[LOG_FUNC_TYPE] = print):
        Timer.__init__(self, prefix=prefix, timer_function=timer_function, log_func=log_func)
        ContinuationDisplayManager.__init__(self)

    def __enter__(self):
        Timer.__enter__(self)
        ContinuationDisplayManager.__enter__(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        Timer.__exit__(self, exc_type, exc_val, exc_tb)
        ContinuationDisplayManager.__exit__(self, exc_type, exc_val, exc_tb)

