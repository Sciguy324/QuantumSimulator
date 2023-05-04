"""
This file contains the main simulator class for running simulations
"""
# Import modules
import numpy as np
from math import factorial

# All declarations to be enabled via from Simulator import *
__all__ = ["Simulation"]


class Simulation:
    """
    Primary simulator class for the
    """

    def __init__(self, dimensions: int, dt: float, hbar: float = 1.0):
        self._time = 0
        self._dt = float(dt)
        self.hbar = float(hbar)
        self._dimensions = int(dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def dt(self) -> float:
        return self._dt
