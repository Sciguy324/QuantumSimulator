# Import modules
from QSim.Simulator import Simulation
from QSim.Render import GLRender1D
from QSim.Math import laplacian
import numpy as np

# Declare constants
hbar = 1
L = 1
m = 1


# Declare hamiltonian for the square well
def H(psi: np.ndarray, x: np.ndarray, dx: float):
    return -hbar**2 / (2*m) * laplacian(psi, (dx,))


def boundaryConditions(psi: np.ndarray, x: np.ndarray, dx: float):
    """End points are zero"""
    psi[0] = 0.0
    psi[-1] = 0.0
    return psi


# Create simulator
simulation = Simulation(np.linspace(0, L, 50, dtype=float),
                        hamiltonian=H, boundary_conditions=boundaryConditions,
                        dt=5e-3, order=70)
simulation.setStateFromFunction(lambda x: np.sqrt(2/L)*np.sin(np.pi*x/L) + np.sqrt(2/L)*np.sin(2*np.pi*x/L))
simulation.step()

# Create renderer
win = GLRender1D()
win.attachSimulation(simulation)
win.start()
