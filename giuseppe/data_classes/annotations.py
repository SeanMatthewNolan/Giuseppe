from typing import Optional
from dataclasses import dataclass


@dataclass
class Annotations:
    independent: Optional[str] = None
    states: Optional[list[str]] = None
    parameters: Optional[list[str]] = None
    constants: Optional[list[str]] = None

    controls: Optional[list[str]] = None

    costates: Optional[list[str]] = None
    initial_adjoints: Optional[list[str]] = None
    terminal_adjoints: Optional[list[str]] = None
