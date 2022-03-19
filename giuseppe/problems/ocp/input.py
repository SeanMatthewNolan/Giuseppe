from dataclasses import dataclass

from giuseppe.problems.bvp.input import InputBVP


@dataclass
class InputCost:
    initial: str = '0'
    path: str = '0'
    terminal: str = '0'


class InputOCP(InputBVP):
    """
    Class to input optimal control problem data for symbolic processing
    """

    def __init__(self):
        super().__init__()

        self.controls = []
        self.cost: InputCost = InputCost()

    def add_control(self, name: str):
        self.controls.append(name)
        return self

    def set_cost(self, initial: str, path: str, terminal: str):
        self.cost = InputCost(initial, path, terminal)
        return self
