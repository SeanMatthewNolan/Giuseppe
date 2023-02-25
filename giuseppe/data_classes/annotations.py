from typing import Optional, Hashable, Sequence
from dataclasses import dataclass


@dataclass
class Annotations:
    independent: Optional[Hashable] = None
    states: Optional[Sequence[Hashable]] = None
    parameters: Optional[Sequence[Hashable]] = None
    constants: Optional[Sequence[Hashable]] = None

    controls: Optional[Sequence[Hashable]] = None

    costates: Optional[Sequence[Hashable]] = None
    initial_adjoints: Optional[Sequence[Hashable]] = None
    terminal_adjoints: Optional[Sequence[Hashable]] = None
