from dataclasses import dataclass


@dataclass
class InputState:
    name: str
    eom: str


@dataclass
class InputConstant:
    name: str
    default_value: float = 0.


@dataclass
class InputCost:
    initial: str = '0'
    path: str = '0'
    terminal: str = '0'


class InputConstraints:
    def __init__(self):
        self.initial: list[str] = []
        self.terminal: list[str] = []


class InputOCP:
    """
    Class to input problem data for symbolic processing
    """
    def __init__(self):
        self.independent = None
        self.states: list[InputState] = []
        self.controls = []
        self.constants: list[InputConstant] = []

        self.cost: InputCost = InputCost()

        self.constraints: InputConstraints = InputConstraints()

    def set_independent(self, var_name: str):
        self.independent = var_name
        return self

    def add_state(self, name: str, eom: str):
        self.states.append(InputState(name, eom))
        return self

    def add_control(self, name: str):
        self.controls.append(name)
        return self

    def add_constant(self, name: str, default_value: float = 0.):
        self.constants.append(InputConstant(name, default_value))
        return self

    def add_constraint(self, location: str, expr: str):
        self.constraints.__getattribute__(location).append(expr)
        return self

    def set_cost(self, initial: str, path: str, terminal: str):
        self.cost = InputCost(initial, path, terminal)
        return self
