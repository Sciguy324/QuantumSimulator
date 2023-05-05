# Import modules
from QSim.Simulator import Simulation
from QSim.Render import GLRender1D, RenderMode
from QSim.Math import laplacian
from scipy.special import hermite
import numpy as np

# Declare constants
hbar = 1
L = 1
m = 1
k = 4

omega = np.sqrt(k/m)
alpha = m*omega/hbar


# Declare hamiltonian for the square well
def V(x):
    return 0.5*k*x**2


def H(psi: np.ndarray, x: np.ndarray, dx: float):
    return -hbar**2 / (2*m) * laplacian(psi, (dx,)) + V(x) * psi


def basisState(x, n):
    return hermite(n)(np.sqrt(alpha)*x)*np.exp(-alpha*x**2/2)


# Create simulator
simulation = Simulation(np.linspace(-3*L, 3*L, 4*50, dtype=float),
                        hamiltonian=H,
                        dt=5e-3, order=70)

simulation.setStateFromFunction(lambda x: basisState(x, 1) + basisState(x, 2))

# Create renderer
win = GLRender1D(width=960, height=540)
win.attachSimulation(simulation)

# Add line to renderer
win.addCurve(lambda x: V(0.3*x), (255, 255, 0))
win.setExtent(xlo=-3*L, xhi=3*L, ylo=0.0, yhi=1.0)
win.setRenderMode(RenderMode.SQUARE_MODULUS)

# Begin
win.start()
