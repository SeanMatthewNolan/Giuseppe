from typing import Protocol, runtime_checkable

import numpy as np

from .ocp import OCP
from .adjoints import Adjoints


@runtime_checkable
class Dual(OCP, Adjoints, Protocol):
    ...
