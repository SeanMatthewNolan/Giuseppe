from giuseppe.io import Solution
from giuseppe.utils.visual.tk_elements.data_viewer import TKDataViewer


class TKSolViewer(TKDataViewer):
    def __init__(self, sol: Solution, master=None):
        super().__init__(master=master)
        self.sol = sol
