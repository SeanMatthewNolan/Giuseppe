from typing import Optional


class Annotations:
    def __init__(self):
        self.independent: Optional[str] = None
        self.states: list[str] = []
        self.parameters: list[str] = []
        self.dynamics: list[str] = []
        self.constants: list[str] = []

        self.costates: list[str] = []
        self.initial_adjoints: list[str] = []
        self.terminal_adjoints: list[str] = []

        self.expressions: list[str] = []
