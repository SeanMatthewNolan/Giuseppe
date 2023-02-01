from typing import Protocol, runtime_checkable

from .ocp import OCP


@runtime_checkable
class CombinedProb(Protocol):
    num_states: int
    num_controls: int
    num_parameters: int
    num_adjoint_parameters: int
    num_constants: int

    primal_problem: OCP
    dual_problem: OCP
