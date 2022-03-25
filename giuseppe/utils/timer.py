from typing import Union, Optional
from collections.abc import Callable

from time import perf_counter_ns

_TIMER_OUTPUT_TYPE = Union[int, float]
TIMER_TYPE = Callable[[], _TIMER_OUTPUT_TYPE]
LOG_FUNC_TYPE = Callable[[str], None]


def format_time(elasped_time: _TIMER_OUTPUT_TYPE):
    if isinstance(elasped_time, int):
        microseconds, nanoseconds = divmod(elasped_time, 1e3)
        milliseconds, microseconds = divmod(microseconds, 1e3)
        seconds, milliseconds = divmod(milliseconds, 1e3)

        if seconds < 10:
            if seconds > 0:
                return '{0:n} s : {1:n} ms'.format(seconds, milliseconds)
            elif milliseconds > 0:
                return '{0:n} ms : {1:n} \u03BCs'.format(milliseconds, microseconds)
            elif microseconds > 0:
                return '{0:n} \u03BCs : {1:n} ns'.format(microseconds, nanoseconds)
            else:
                return '{0:n} ns'.format(nanoseconds)

        elasped_time /= 1e9

    minutes, seconds = divmod(elasped_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    if days > 0:
        return '{0:n} day(s), {1:n} hr, {2:n} min'.format(days, hours, minutes)
    elif hours > 0:
        return '{0:n} hr, {1:n} min, {2:n} s'.format(hours, minutes, seconds)
    elif minutes > 0:
        return '{0:n} min : {1:0.3f} s'.format(minutes, seconds)
    else:
        return '{0:0.3f} s'.format(seconds)


class Timer:
    def __init__(self, prefix: str = '', timer_function: TIMER_TYPE = perf_counter_ns,
                 log_func: Optional[LOG_FUNC_TYPE] = print):
        self.timer: TIMER_TYPE = timer_function
        self.start_time: _TIMER_OUTPUT_TYPE = -1
        self.end_time: _TIMER_OUTPUT_TYPE = -1
        self.elasped_time: _TIMER_OUTPUT_TYPE = 0

        self.prefix: str = prefix + ' '

        self.log_fun: Optional[LOG_FUNC_TYPE] = log_func

    def __enter__(self):
        self.start_time = self.timer()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_time = self.timer()
        self.elasped_time = self.end_time - self.start_time
        if self.log_fun is not None:
            self.log_fun(f'{self.prefix}Elasped time: {format_time(self.elasped_time)}')

        return False
