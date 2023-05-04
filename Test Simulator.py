# Import modules
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time_ns
from math import factorial


# Declare constants
m = 1
hbar = 1
dt = 5e-3

# Note:
# aΔt/Δx² < 0.5


# Declare some helper functions
def laplacian1D(psi, deltas):
    """Computes the Laplacian of a function"""
    result = np.zeros(psi.shape, dtype='complex128')
    result[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / deltas[0]**2
    return result


def laplacian(psi, deltas):
    """Computes the Laplacian of a function"""
    if len(deltas) == 1:
        return laplacian1D(psi, deltas)
        
    result = np.zeros(psi.shape, dtype=float)
    maxDim = len(psi.shape)
    for axis, delta in zip(range(maxDim-1, -1, -1), deltas):
        result = result + np.gradient(np.gradient(psi, axis=axis), axis=axis) / delta
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


def normalize(psi, deltas):
    """Normalzes the given wavefunction"""
    a = np.sqrt(nquad(np.conjugate(psi)*psi, deltas))
    return psi / a


# Declare simulation evolution rule
def step(psi, hamiltonian, deltas, order=1):
    # Output buffer
    result = psi.copy()

    # Apply N orders of approximation
    for n in range(1, order+1):
        # Coefficient out front
        c = 1/factorial(n)*(-1j*dt/hbar)**n
        # Apply hamiltonian N times
        buffer = psi.copy()
        for i in range(n):
            buffer = hamiltonian(buffer, deltas)
        # Add to final result
        result = result + c*buffer
        
    
##    # Laplacian term
##    c1 = -dt/1j*hbar/(2*m) * laplacian(psi, deltas)
##
##    # Potential term
##    c2 = (dt / (1j*hbar)*V+1)*psi
##
##    # Return result
##    return c1 + c2

    return result


def H(psi, deltas):
    # Apply Laplacian term
    result = -hbar**2 / (2*m) * laplacian(psi, deltas)

    # Apply potential term
    # No potential
    return result


# Declare boundary conditions
def boundaryConditions(psi):
    psi[0] = 0
    psi[-1] = 0
    return psi


L = 1
x = np.linspace(0, L, 50)
dx = x[1] - x[0]

print(dt/dx**2)

# Declare potential: infinite square well
V = np.zeros(x.shape, dtype=float)

# Declare initial wavefunction
psi = np.sqrt(2/L)*np.sin(np.pi*x/L) + np.sqrt(2/L)*np.sin(3*np.pi*x/L)



# Generate animated plot of simulation
fig, ax = plt.subplots()
line1, = ax.plot(x, np.real(psi), label=r'Re($\psi$)', animated=True)
#line2, = ax.plot(x, np.imag(psi), label=r'Im($\psi$)', animated=True)
line3, = ax.plot(x, np.conjugate(psi)*psi, label=r'$|\psi|^2$', animated=True)
#ax.legend()
max_y = 4
min_y = -2
bg = fig.canvas.copy_from_bbox(fig.bbox)

plt.show(block=False)
plt.pause(0.1)

while True:
    # Perform calculations
    psi = step(psi, H, (dx,), order=70)
    psi = boundaryConditions(psi)
    psi = normalize(psi, (dx,))
    # Redraw plot
    fig.canvas.draw()
    # Update lines
    line1.set_ydata(np.real(psi))
    line3.set_ydata(np.conjugate(psi)*psi)
    ax.draw_artist(line1)
    ax.draw_artist(line3)
    # Update figure
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()

##def animate(i):
##    global psi
##    global ax
##    global max_y, min_y
##    psi_before = psi.copy()
##    psi = step(psi, H, (dx,))
##    psi = boundaryConditions(psi)
##    #print(np.max(np.abs(psi-psi_before)))
##    #psi = normalize(psi, (dx,))
##    line1.set_ydata(np.real(psi))
##    line2.set_ydata(np.imag(psi))
##    max_y = max(max_y, np.max(np.real(psi)))
##    min_y = min(min_y, np.min(np.real(psi)))
##    ax.set_ylim(min_y, max_y)
##    line3.set_ydata(np.conjugate(psi)*psi)
##    return line1, line2, line3
##
##ani = animation.FuncAnimation(fig, animate, interval=1, blit=True, save_count=50)
##fig.show()

