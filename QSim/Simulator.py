"""
This file contains the main simulator class for running simulations
"""
# Import modules
import numpy as np
from math import factorial
from .Wavefunction import AbstractState
from typing import Callable, Union
from time import time

# All declarations to be enabled via from Simulator import *
__all__ = ["Simulation"]


class Simulation:
    """
    Primary simulation class for the quantum simulator
    """

    def __init__(self, hamiltonian: Callable, dt: float, hbar: float = 1.0, order=50):
        self._time: int = 0
        self.dt: float = float(dt)
        self.hbar: float = float(hbar)
        self.hamiltonian: Callable = hamiltonian
        self.order = order
        self._state: Union[AbstractState, None] = None

    @property
    def time(self) -> float:
        """Returns the timestep"""
        return self._time

    @property
    def state(self) -> Union[None, AbstractState]:
        return self._state.copy()

    def setHamiltonian(self, hamiltonian: Callable):
        self.hamiltonian = hamiltonian

    def setState(self, state: AbstractState):
        """Sets the internal wavefunction of the simulator"""
        self._state = state

    def step(self):
        """Runs a single iteration of the simulator"""
        # Increment timer
        self._time += self.dt

        # Output buffer
        if self._state is None:
            raise ValueError("No state vector has been set")
        result = self._state.copy()

        # Apply N orders of approximation
        print('-'*20)
        for n in range(1, self.order + 1):
            t1 = time()
            # Coefficient out front
            c = 1 / factorial(n) * (-1j * self.dt / self.hbar) ** n
            # Apply hamiltonian N times
            buffer = self._state.copy()
            t3 = time()
            for i in range(n):
                buffer = self.hamiltonian(buffer)
            t4 = time()
            # Add to final result
            result = result + c * buffer
            t2 = time()
            print(t2-t1, t4-t3)

        # Normalize and return result
        result.normalize()
        return result
