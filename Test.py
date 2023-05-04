# Import modules
from QSim.Simulator import Simulation
from QSim.Wavefunction import ContinuousState
from QSim.Render import GLRender1D
import numpy as np

# Declare constants
hbar = 1
L = 1
m = 1

# Create state vector
state = ContinuousState(np.linspace(0, L, 40))
state.setFromFunc(lambda x: np.sqrt(2/L)*np.sin(np.pi*x/L) + np.sqrt(2/L)*np.sin(3*np.pi*x/L))


# Declare hamiltonian
def H(psi: ContinuousState):
    return -hbar**2 / (2*m) * psi.laplacian()


# Create simulator
simulation = Simulation(H, 5e-3)
simulation.setState(state)
simulation.step()

# Create renderer
win = GLRender1D()
win.attachSimulation(simulation)
win.start()
