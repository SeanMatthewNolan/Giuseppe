from contextlib import AbstractContextManager


class ContinuationMonitor(AbstractContextManager):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def start_cont_series(self, series):
        pass

    def log_step(self):
        pass

    def log_msg(self, msg: str):
        pass

    def end_cont_series(self):
        pass

