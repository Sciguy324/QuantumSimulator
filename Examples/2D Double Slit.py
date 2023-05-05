# Import modules
from QSim.Simulator import Simulation
from QSim.Render import GLRender2D
from QSim.Math import laplacian
import numpy as np

# Declare constants
hbar = 1
L = 2
m = 1
beta = 1
p0 = -5


def V(x, y):
    barrier = ((x < -0.2*L) & (-0.1*L < y) & (y < 0.1*L)) | ((x > 0.2*L) & (-0.1*L < y) & (y < 0.1*L)) | ((-0.05*L < x) & (x < 0.05*L) & (-0.1*L < y) & (y < 0.1*L))
    return barrier.astype(float)*100.0


def boundary_conditions(psi, x, y, dx, dy):
    psi[0] = 0
    psi[-1] = 0
    psi[:, 0] = 0
    psi[:, -1] = 0
    return psi


# Declare hamiltonian for the square well
def H(psi: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, dy: float):
    return -hbar**2 / (2*m) * laplacian(psi, (dx, dy)) + V(x, y)*psi


def initialState(x, y):
    r = np.sqrt(((x + 0.0) ** 2 + (y - 0.5*L) ** 2))
    return np.exp(1j*p0*y/hbar)*np.exp(-r**2*beta/hbar**2)


# Create simulator
simulation = Simulation(np.linspace(-L, L, 50, dtype=float),
                        np.linspace(-L, L, 50, dtype=float),
                        hamiltonian=H, boundary_conditions=boundary_conditions,
                        dt=5e-3, order=50)
simulation.setStateFromFunction(initialState)

# Create renderer
win = GLRender2D(width=500, height=500)
win.attachSimulation(simulation)
win.setExtent(xlo=-L, xhi=L, ylo=-L, yhi=L)
win.setColorRange(0.0, 1.0)
win.addBox(-0.1*L, -L, 0.1*L, -0.2*L, (0, 0, 0))
win.addBox(-0.1*L, 0.2*L, 0.1*L, L, (0, 0, 0))
win.addBox(-0.1*L, -0.1*L, 0.1*L, 0.1*L, (0, 0, 0))
win.paused = True
win.start()
