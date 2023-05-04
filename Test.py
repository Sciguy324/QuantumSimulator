# Import modules
from QSim.Simulator import Simulation
from QSim.Render import GLRender1D, GLRender2D
from QSim.Math import laplacian
import numpy as np

# Declare constants
hbar = 1
L = 1
m = 1

TEST = '2D-Potential'

if TEST == '1D':
    # Declare hamiltonian (square well)
    def H(psi: np.ndarray, x: np.ndarray):
        dx = x[1] - x[0]
        return -hbar**2 / (2*m) * laplacian(psi, (dx,))


    # Create simulator
    simulation = Simulation(np.linspace(0, L, 50, dtype=float),
                            hamiltonian=H, dt=5e-3, order=70)
    simulation.setStateFromFunction(lambda x: np.sqrt(2/L)*np.sin(np.pi*x/L) + np.sqrt(2/L)*np.sin(2*np.pi*x/L))
    simulation.step()

    # Create renderer
    win = GLRender1D()
    win.attachSimulation(simulation)
    win.start()


if TEST == '2D':
    # Declare hamiltonian (square well)
    def H(psi: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, dy: float):
        return -hbar ** 2 / (2 * m) * laplacian(psi, (dx, dy)) + psi

    def basisState(x, y, nx, ny):
        return np.sin(np.pi * nx * x / L) * np.sin(np.pi * ny * y / L)

    # Create simulator
    simulation = Simulation(np.linspace(0, L, 20, dtype=float),
                            np.linspace(0, L, 20, dtype=float),
                            hamiltonian=H, dt=5e-3, order=30)
    simulation.setStateFromFunction(lambda x, y: basisState(x, y, 1, 1)+basisState(x, y, 2, 2))
    simulation.step()

    # Create renderer
    win = GLRender2D(width=500, height=500)
    win.attachSimulation(simulation)
    win.start()


if TEST == '2D-Potential':
    a = 500
    k = 10

    # Declare hamiltonian (square well)
    def H(psi: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, dy: float):
        r = np.sqrt(x**2 + y**2)
        return -hbar ** 2 / (2 * m) * laplacian(psi, (dx, dy)) + k/(r+0.1)*psi

    def basisState(x, y):
        r = np.sqrt(x**2 + y**2)
        #theta = np.arctan2(x, y)
        R = r/a*np.exp(-0.5*r/a)
        #Theta = np.sin(theta)
        return R#*Theta

    def basisState(x, y, nx=1, ny=1):
        return np.cos(np.pi * nx * x / L) * np.cos(np.pi * ny * y / L)

    # Create simulator
    simulation = Simulation(np.linspace(-L/2, L/2, 20, dtype=float),
                            np.linspace(-L/2, L/2, 20, dtype=float),
                            hamiltonian=H, dt=5e-3, order=30)
    simulation.setStateFromFunction(basisState)
    simulation.step()

    # Create renderer
    win = GLRender2D(width=500, height=500)
    win.attachSimulation(simulation)
    win.start()

