from typing import Protocol

from .ocp import OCP


class CombinedProb(Protocol):
    num_states: int
    num_controls: int
    num_parameters: int
    num_adjoint_parameters: int
    num_constants: int

    primal_problem: OCP
    dual_problem: OCP
