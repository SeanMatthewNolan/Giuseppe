class InputSymbolicProblem:
    """
    Class to input problem data for symbolic processing
    """
    def __init__(self):
        self.independent = None
        self.states = []
        self.controls = []
        self.constants = []

        self.cost = {
            'initial': '0',
            'path': '0',
            'terminal': '0',
        }

        self.constraints = {
            'initial': [],
            'terminal': [],
        }

    def set_independent(self, var_name: str):
        self.independent = var_name
        return self

    def add_state(self, name: str, eom: str):
        self.states.append({'name': name, 'eom': eom})
        return self

    def add_control(self, name: str):
        self.controls.append(name)
        return self

    def add_constant(self, name: str, default_value: float = -1.):
        self.constants.append({'name': name, 'default_value': default_value})
        return self

    def add_constraint(self, location: str, expr: str):
        self.constraints[location].append(expr)
        return self

    def set_cost(self, initial: str, path: str, terminal: str):
        self.cost['initial'] = initial
        self.cost['path'] = path
        self.cost['initial'] = terminal
        return self
