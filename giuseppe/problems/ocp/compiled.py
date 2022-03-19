from copy import deepcopy

from .symbolic import SymOCP


class CompOCP:
    def __init__(self, source_ocp: SymOCP):
        src_ocp = deepcopy(source_ocp)  # source bvp is copied here for reference as it may be mutated later
        self.src_ocp = src_ocp

        if isinstance(source_ocp, SymOCP):
            self._sym_args = {
                'boundary': [src_ocp.independent, src_ocp.states, src_ocp.constants],
                'dynamic': [src_ocp.independent, src_ocp.states, src_ocp.controls, src_ocp.constants]
            }

        else:
            raise NotImplementedError(f'CompOCP cannot ingest type {type(source_ocp)}')
