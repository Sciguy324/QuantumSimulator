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
k = 5
n = 2

omega = np.sqrt(k/m)
alpha = m*omega/hbar
print(hbar*omega*(n+0.5))


# Declare hamiltonian for the square well
def V(x):
    return 0.5*k*x**2


def H(psi: np.ndarray, x: np.ndarray, dx: float):
    return -hbar**2 / (2*m) * laplacian(psi, (dx,)) + V(x) * psi


# def boundaryConditions(psi: np.ndarray, x: np.ndarray, dx: float):
#     """Repeat values"""
#     psi[0] = psi[1]
#     psi[-1] = psi[-2]
#     return psi

def boundaryConditions(psi: np.ndarray, x: np.ndarray, dx: float):
    """Repeat values"""
    psi[0] = 0.0
    psi[-1] = 0.0
    return psi


# Create simulator
simulation = Simulation(np.linspace(-3*L, 3*L, 4*50, dtype=float),
                        hamiltonian=H, #boundary_conditions=boundaryConditions,
                        dt=5e-3, order=70)
poly = hermite(n)
simulation.setStateFromFunction(lambda x: poly(np.sqrt(alpha)*x)*np.exp(-alpha*x**2/2))

# Create renderer
win = GLRender1D(width=960, height=540)
win.attachSimulation(simulation)

# Add line to renderer
win.addCurve(lambda x: V(0.3*x), (255, 255, 0))
win.setExtent(xlo=-3*L, xhi=3*L, ylo=0.0, yhi=1.0)
win.setRenderMode(RenderMode.SQUARE_MODULUS)

# Begin
win.start()
