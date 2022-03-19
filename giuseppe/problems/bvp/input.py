from dataclasses import dataclass


@dataclass
class InputState:
    name: str
    eom: str


@dataclass
class InputConstant:
    name: str
    default_value: float = 0.


class InputConstraints:
    def __init__(self):
        self.initial: list[str] = []
        self.terminal: list[str] = []


class InputBVP:
    """
    Class to input boundary value problem data for symbolic processing
    """
    def __init__(self):
        self.independent = None
        self.states: list[InputState] = []
        self.constants: list[InputConstant] = []
        self.constraints: InputConstraints = InputConstraints()

    def set_independent(self, var_name: str):
        self.independent = var_name
        return self

    def add_state(self, name: str, eom: str):
        self.states.append(InputState(name, eom))
        return self

    def add_constant(self, name: str, default_value: float = 0.):
        self.constants.append(InputConstant(name, default_value))
        return self

    def add_constraint(self, location: str, expr: str):
        self.constraints.__getattribute__(location).append(expr)
        return self
