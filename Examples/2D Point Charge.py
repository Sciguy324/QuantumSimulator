# Import modules
from QSim.Simulator import Simulation
from QSim.Render import GLRender2D
from QSim.Math import laplacian
import numpy as np

# Declare constants
hbar = 1
L = 1
m = 1
a = 1.0


# Declare hamiltonian for the square well
def H(psi: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, dy: float):
    r = np.sqrt(x ** 2 + y ** 2)
    return -hbar**2 / (2*m) * laplacian(psi, (dx, dy)) - 5/(r+0.001)*psi


def basisState(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(x, y)
    #R = (2-r / a) * np.exp(-0.5*r / a)
    R = np.exp(-0.5*r / a)*np.cos(theta)
    return R


# Create simulator
simulation = Simulation(np.linspace(-3*L, 3*L, 4*20, dtype=float),
                        np.linspace(-3*L, 3*L, 4*20, dtype=float),
                        hamiltonian=H,
                        dt=5e-3, order=30)
simulation.setStateFromFunction(lambda x, y: basisState(x, y))

# Create renderer
win = GLRender2D(width=500, height=500)
win.attachSimulation(simulation)
win.setExtent(xlo=-2*L, xhi=2*L, ylo=-2*L, yhi=2*L)
win.setColorRange(0.0, 1.0)
win.start()
