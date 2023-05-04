"""
This file contains the main simulator class for running simulations
"""
# Import modules
import numpy as np
from math import factorial
from typing import Callable, Tuple
from .Math import nquad

# All declarations to be enabled via from Simulator import *
__all__ = ["Simulation"]


# Declare helper functions
class Simulation:
    """
    Primary simulation class for the quantum simulator
    """

    def __init__(self, *axes, hamiltonian: Callable = None, dt: float = 0.1, hbar: float = 1.0, order=50):
        """
        :param axes: A set of 1D numpy arrays defining the axes of the system.  Expected to be evenly spaced
        :param hamiltonian: Function used to compute the wavefunction's energy (defaults to 0)
        :param dt: Timestep per tick (defaults to 0.1)
        :param hbar: Planck's constant (defaults to 1)
        :param order: Correction order to compute each step with (defaults to order 50)
        """
        # Build meshgrid
        self._deltas = tuple([float(np.mean(np.diff(i))) for i in axes])
        self._meshgrid: Tuple[np.ndarray] = tuple(np.meshgrid(*axes))

        # Declare internal state
        self._psi: np.ndarray = np.ones(self._meshgrid[0].shape, dtype='complex128')
        self.normalize()

        # Set parameters
        self._step_count: int = 0
        self._time: float = 0
        self._auto_normalize = True
        self.dt: float = float(dt)
        self.hbar: float = float(hbar)
        if hamiltonian is None:
            self.hamiltonian = lambda x, *meshgrid, delta: x*0
        else:
            self.hamiltonian: Callable = hamiltonian
        self.order = order

    @property
    def dims(self) -> int:
        """The number of spacial dimensions in the simulation"""
        return np.ndim(self._psi)

    @property
    def deltas(self) -> Tuple[float]:
        """The spatial steps in the internal meshgrid"""
        return self._deltas

    @property
    def psi(self) -> np.ndarray:
        """Returns a COPY of the internal wavefunction"""
        return self._psi

    @property
    def squareMod(self) -> np.ndarray:
        """Returns the square-modulus of the internal wavefunction"""
        return np.real(np.conjugate(self._psi)*self._psi)

    @property
    def meshgrid(self) -> Tuple[np.ndarray]:
        """Returns a COPY of the internal wavefunction"""
        return self._meshgrid

    @property
    def time(self) -> float:
        """Returns the timestep"""
        return self._time

    def enableNormalization(self):
        """Enables auto-normalization of the internal wavefunction"""
        self._auto_normalize = True

    def disableNormalization(self):
        """Enables auto-normalization of the internal wavefunction"""
        self._auto_normalize = False

    def setHamiltonian(self, hamiltonian: Callable):
        """Sets the hamiltonian to the provided function"""
        self.hamiltonian = hamiltonian

    def setStateFromArray(self, array: np.ndarray):
        """
        Sets the internal wavefunction of the simulator.

        :param array: Array representing the new value of psi.  Shape must match the internal meshgrid.
        """
        # Enforce type
        array = np.array(array, dtype='complex128')

        # Shape check
        if self._meshgrid[0].shape != array.shape:
            raise ValueError(f"Shape mismatch between provided array and internal meshgrid.  Received {array.shape}, "
                             f"expected {self._meshgrid[0].shape}")

        # Pass to internal state
        self._psi = array

        # Normalize
        self.normalize()

    def setStateFromFunction(self, function: Callable):
        """
        Sets the internal wavefunction by evaluating the provided function over the internal meshgrid

        :param function: Function to evaluate
        """
        # Compute function
        array = function(*self.meshgrid)

        # Assign to internal array
        self.setStateFromArray(array)

    def normalize(self):
        """Normalizes the internal wavefunction"""
        self._psi = self.normalized()

    def normalized(self) -> np.ndarray:
        """Returns a normalized copy of the internal wavefunction"""
        a = np.sqrt(nquad(self.squareMod, self.deltas))
        return self._psi / a

    def step(self):
        """Runs a single iteration of the simulator"""
        # Increment timer
        self._time += self.dt
        self._step_count += 1

        # Output buffer
        result = self.psi

        # Apply N orders of approximation
        for n in range(1, self.order + 1):
            # Coefficient out front
            c = 1 / factorial(n) * (-1j * self.dt / self.hbar) ** n

            # Apply hamiltonian N times
            buffer = self.psi.copy()
            for i in range(n):
                buffer = self.hamiltonian(buffer, *self.meshgrid, *self.deltas)

            # Add to final result
            result = result + c * buffer

        # Pass result back into internal wavefunction
        self._psi = result

        # Normalize if applicable
        if self._auto_normalize:
            self.normalize()
