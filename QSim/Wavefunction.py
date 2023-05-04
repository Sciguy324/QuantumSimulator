"""
This module contains classes/methods to represent the state of a wavefunction
"""
# Import modules
import numpy as np
from typing import Callable, Union, Iterable
from numbers import Number

__all__ = ["ContinuousState", "DiscreteState"]


class AbstractState:
    """
    Class representing the state of a wavefunction
    """

    def __init__(self, basis):
        """
        Constructor

        :param basis: The labels for the basis states.  Must be a numpy array of any shape.
        """
        self._basis = basis
        self._state: np.ndarray = np.array([], dtype='complex128')

    @property
    def basis(self):
        return self._basis.copy()

    @property
    def dims(self) -> int:
        return len(self._basis)

    def toArray(self):
        """Returns a copy of the internal state as a numpy array"""
        return self._state.copy()

    def _checkAgainstBasis(self, array: np.ndarray):
        """Verifies that the provided array is valid against the internal basis.  Override in subclass."""
        raise NotImplementedError()

    def setState(self, psi: np.ndarray, normalize=True):
        """
        Sets the internal state vector to the provided array.

        :param psi: A state vector represented as a numpy array.  Size must match the internal basis.
        :param normalize: Whether to immediately normalize the state vector.  Defaults to true.
        """
        # Enforce type
        psi = np.array(psi, dtype='complex128')
        # Check shape
        if not self._checkAgainstBasis(psi):
            raise ValueError(f"Shape mismatch between new state and basis states.  Basis state array has dimensions "
                             f"of {self._basis[0].shape}, while new state has dimensions of {psi.shape}")
        self._state = psi

        # Apply normalization
        if normalize:
            self.normalize()

    def setFromFunc(self, function: Callable, normalize=True):
        """Sets the internal state by providing the internal basis to the given function"""
        self._state = function(self._basis)
        if normalize:
            self.normalize()

    def __mul__(self, other):
        if isinstance(other, Number):
            # Scalar multiplication
            new_state = self.copy()
            new_state._state = new_state._state * other
            return new_state
        else:
            # Inner product
            return self.innerProduct(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        new_state = self.copy()
        if isinstance(other, AbstractState):
            other_processed = other._state
        else:
            # F*ck around and find out
            other_processed = other

        new_state.setState(self._state + other_processed)
        return new_state

    def innerProduct(self, other) -> complex:
        """
        Override in subclass
        """
        raise NotImplementedError()

    def normalize(self):
        """Normalizes the wavefunction"""
        N = self.innerProduct(self)
        self._state = self._state / np.sqrt(N)

    def copy(self):
        """Override in subclass"""
        raise NotImplementedError()


class ContinuousState(AbstractState):
    """
    Class representing states with a 'continuous' basis
    """
    def __init__(self, *axes: np.ndarray):
        super().__init__(np.meshgrid(*axes))
        self._state = np.ones(self._basis[0].shape, dtype='complex128')
        self.normalize()

    def _checkAgainstBasis(self, array: np.ndarray):
        return array.shape == self._basis[0].shape

    def innerProduct(self, other) -> complex:
        """
        Takes the inner product between this state vector and another
        :param other: Discrete state to take dot product with
        """
        # Type check
        if not isinstance(other, ContinuousState):
            raise TypeError(f"Cannot take inner product between {type(self)} and {type(other)}")

        # Perform integration and return result
        result = nquad(np.conjugate(self.toArray()) * other.toArray(), self._basis)
        return result

    def laplacian1D(self):
        """Computes the Laplacian of a 1D function"""
        delta = np.gradient(self._basis[0], axis=0)
        result = np.gradient(np.gradient(self._state, axis=0) / delta, axis=0) / delta
        new_state = self.copy()
        new_state.setState(result)
        return new_state

    def laplacian(self):
        """Computes the Laplacian of a function"""
        if self.dims == 1:
            return self.laplacian1D()
        result = np.zeros(self._state.shape, dtype='complex128')
        for axis in range(self.dims - 1, -1, -1):
            delta = np.gradient(self._basis[axis], axis=axis)
            result = result + np.gradient(np.gradient(self._state, axis=axis) / delta, axis=axis) / delta
        new_state = self.copy()
        new_state.setState(result)
        return new_state

    def copy(self):
        """Returns a copy of the state"""
        new_state = ContinuousState(self.basis)
        new_state.setState(self.toArray())
        return new_state

    def setFromFunc(self, function: Callable, normalize=True):
        """Sets the internal state by providing the internal basis to the given function"""
        self._state = function(*self._basis)
        if normalize:
            self.normalize()


class DiscreteState(AbstractState):
    """
    Class representing states with a discrete basis
    """
    def __init__(self, labels: Union[Iterable, np.ndarray]):
        labels = np.array(labels)
        super().__init__(labels)
        self._state = np.ones(labels.shape)

    def __matmul__(self, other):
        new_state = self.copy()
        new_state._state = other @ new_state._state
        return new_state

    def print(self, fmt: str = '.3f') -> str:
        """
        Creates a string representation of this state as a linear combination of kets

        :param fmt: Formatting code for the coefficients.
        :return: String representation of state
        """
        # Check if this is even possible
        if np.ndim(self._state) > 1:
            NotImplementedError("Printing not implemented for ")
        # Compile string
        result = []
        for label, value in zip(self._basis, self._state):
            if np.imag(value) == 0.0:
                result.append(f'{np.real(value):{fmt}}|{label}〉')
            elif np.real(value) == 0.0:
                result.append(f'{np.imag(value):{fmt}}j|{label}〉')
            else:
                result.append(f'{value:{fmt}}j|{label}〉')

        return "+ ".join(result)

    def _checkAgainstBasis(self, array: np.ndarray):
        """Verifies that the provided array is valid against the internal basis."""
        return array.shape == self._basis.shape

    def innerProduct(self, other) -> complex:
        """
        Takes the inner product between this state vector and another
        :param other: Discrete state to take dot product with
        """
        # Type check
        if not isinstance(other, DiscreteState):
            raise TypeError(f"Cannot take inner product between {type(self)} and {type(other)}")

        # Take standard dot product
        return complex(np.sum(np.conjugate(self._state) * other.toArray()))

    def copy(self):
        """Returns a copy of the state"""
        new_state = DiscreteState(self.basis)
        new_state.setState(self.toArray())
        return new_state


def integrateArray(array: np.ndarray, basis):
    """Integrates an array over a single axis using the trapezoidal method"""
    delta = basis[1:] - basis[:-1]
    return np.sum((array[1:] + array[:-1]) / 2 * delta, axis=0)


def nquad(psi: np.ndarray, bases: list):
    """Integrates an array over all dimensions using the trapezoidal method repeatedly"""
    result = psi.copy()
    for basis in bases:
        result = integrateArray(result, basis)
    return result


# Tests
if __name__ == '__main__':
    print('=========Discrete State Tests========')
    print('Normalization Test:')
    test_state_1 = DiscreteState(np.array([1, 2, 3]))
    print('    Before:', test_state_1.print())
    test_state_1.normalize()
    print('    After:', test_state_1.print())

    print('Setting from function:')
    test_state_2 = DiscreteState(np.array([1, 2, 3]))
    test_state_2.setFromFunc(lambda x: x)
    print('    ', test_state_2.print())

    print('Setting from array and string labels:')
    test_state_2 = DiscreteState(np.array(['00', '01', '10']))
    test_state_2.setState(np.array([1.0, 1.0, 0.0]))
    print('    ', test_state_2.print())

    print('Inner Product:', test_state_1 * test_state_2)
    print('=====================================')

    print('=========Continuous State Tests========')
    test_state_1 = ContinuousState(np.linspace(0, 1, 50))
    print('Laplacian of Equal Everywhere:', test_state_1.laplacian())
    test_state_2 = ContinuousState(np.linspace(0, 1, 50))
    test_state_2.setFromFunc(lambda x: x**2)
    print('Laplacian of x² after normalization', test_state_2.laplacian())
    print('Inner Product (〈ψ₁|ψ₂〉):', test_state_1 * test_state_2)
    print('Inner Product (〈ψ₂|ψ₁〉):', test_state_2 * test_state_1)
    print('Inner Product (〈ψ₁|ψ₁〉):', test_state_1 * test_state_1)
    print('Inner Product (〈ψ₂|ψ₂〉):', test_state_2 * test_state_2)
