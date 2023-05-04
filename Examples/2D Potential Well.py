# Import modules
from QSim.Simulator import Simulation
from QSim.Render import GLRender2D
from QSim.Math import laplacian
import numpy as np

# Declare constants
hbar = 1
L = 1
m = 1


# Declare hamiltonian for the square well
def H(psi: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, dy: float):
    return -hbar**2 / (2*m) * laplacian(psi, (dx, dy))


def boundaryConditions(psi: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, dy: float):
    """Edges points are zero"""
    psi[0] = 0.0
    psi[-1] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0
    return psi


def basisState(x, y, n1, n2):
    return np.sin(np.pi*x*n1/L)*np.sin(np.pi*y*n2/L)


# Create simulator
simulation = Simulation(np.linspace(0, L, 30, dtype=float),
                        np.linspace(0, L, 30, dtype=float),
                        hamiltonian=H, boundary_conditions=boundaryConditions,
                        dt=5e-3, order=50)
simulation.setStateFromFunction(lambda x, y: basisState(x, y, 1, 1) + basisState(x, y, 2, 2))
simulation.step()

# Create renderer
win = GLRender2D(width=500, height=500)
win.attachSimulation(simulation)
win.start()
