from contextlib import AbstractContextManager


class ContinuationMonitor(AbstractContextManager):
    def __init__(self):
        self.current_series = None
        self.step_idx = 0
        self.messages = []

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def start_cont_series(self, series):
        self.current_series = series
        self.step_idx = 0

    def log_step(self):
        self.step_idx += 1

    def log_msg(self, msg: str):
        self.messages.append(msg)
        print(msg)

    def end_cont_series(self):
        pass

