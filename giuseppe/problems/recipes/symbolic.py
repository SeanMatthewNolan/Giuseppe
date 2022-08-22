from giuseppe.problems.dual import SymDual, SymDualOCP, CompDualOCP
from giuseppe.problems.ocp import SymOCP
from giuseppe.utils.timer import Timer

from .base import RecipeBase
from ..ocp import InputOCP


# TODO Develop standardized naming scheme
# TODO Make logging more flexible
class DualizeSymbolic(RecipeBase):
    def __call__(self, input_ocp: InputOCP, control_method: str = 'differential'):
        with Timer('Dualization Time: ', ):
            with Timer('Input OCP Sympified: '):
                sym_ocp = SymOCP(input_ocp)

            with Timer('Dual Problem Formed: '):
                sym_dual = SymDual(sym_ocp)

            with Timer('Control Law Computed: '):
                sym_bvp = SymDualOCP(sym_ocp, sym_dual, control_method=control_method)

            with Timer('Problem Compiled: '):
                comp_dual_ocp = CompDualOCP(sym_bvp)

        return comp_dual_ocp
