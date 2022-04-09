from giuseppe.problems.bvp.input import InputBVP
from giuseppe.problems.components.input import InputCost


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
