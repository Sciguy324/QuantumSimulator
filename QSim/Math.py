"""
This module contains a handful of useful math functions
"""
# Import modules
import numpy as np
from scipy.ndimage import laplace

__all__ = ['laplacian', 'nquad']


def laplacian1D(psi, deltas):
    """Computes the Laplacian of a 1D function"""
    result = np.zeros(psi.shape, dtype='complex128')
    result[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / deltas[0]**2
    return result


def laplacian2D(psi, deltas):
    """Computes the Laplacian of a 1D function"""
    result = np.zeros(psi.shape, dtype='complex128')
    result[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / deltas[0]**2
    result[:, 1:-1] = result[:, 1:-1] + (psi[:, 2:] - 2 * psi[:, 1:-1] + psi[:, :-2]) / deltas[1]**2
    return result


def laplacian3D(psi, deltas):
    """Computes the Laplacian of a 1D function"""
    result = np.zeros(psi.shape, dtype='complex128')
    result[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / deltas[0]**2
    result[:, 1:-1] = result[:, 1:-1] + (psi[:, 2:] - 2 * psi[:, 1:-1] + psi[:, :-2]) / deltas[1]**2
    result[:, :, 1:-1] = result[:, :, 1:-1] + (psi[:, :, 2:] - 2 * psi[:, :, 1:-1] + psi[:, :, :-2]) / deltas[2] ** 2
    return result


def laplacian(psi, deltas):
    """Computes the Laplacian of a function"""
    # Use faster algorithms, if possible
    # 1D laplacian
    if len(deltas) == 1:
        return laplacian1D(psi, deltas)

    # 2D laplacian
    elif len(deltas) == 2:
        return laplacian2D(psi, deltas)

    # All deltas are equal: use scipy's faster function
    if np.all(np.diff(deltas) == 0):
        return laplace(psi) / deltas[0]**2

    # Slow way it is, then
    result = np.zeros(psi.shape, dtype=float)
    maxDim = len(psi.shape)
    for axis, delta in zip(range(maxDim - 1, -1, -1), deltas):
        result = result + np.gradient(np.gradient(psi, axis=axis) / delta, axis=axis) / delta
    return result


def integrateArray(array, delta=1):
    """Integrates an array over a single axis using the trapezoidal method"""
    return np.sum((array[1:] + array[:-1])/2 * delta, axis=0)


def nquad(psi, deltas):
    """Integrates an array over all dimensions using the trapezoidal method repeatedly"""
    dim = len(psi.shape)
    result = psi.copy()
    for delta in deltas:
        result = integrateArray(result, delta=delta)
    return result
