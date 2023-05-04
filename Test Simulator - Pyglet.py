# Import modules
import numpy as np
#import scipy.integrate as integrate
from time import time_ns
from math import factorial
# Import pyglet stuff
import pyglet
import pyglet.gl as gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3, Vec2


# Declare shader programs
vertex_source = """#version 150 core
    in vec2 position;
    in vec3 color;
    out vec4 vertex_color;

    uniform mat4 projection;
    uniform vec2 translate;

    void main()
    {
        // Transform position into viewport space
        gl_Position = projection * vec4(position+translate, 0.0, 1.0);
        vertex_color = vec4(color, 1.0);
    }
"""


fragment_source = """#version 150 core
    in vec4 vertex_color;
    in vec2 fragCoord;
    out vec4 final_color;

    void main()
    {
        final_color = vertex_color;
    }
"""


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

    return result


def H(psi, deltas):
    # Apply Laplacian term
    result = -hbar**2 / (2*m) * laplacian(psi, deltas)

    # Apply potential term
    result = result + psi*V
    # No potential
    return result


# Declare boundary conditions
def boundaryConditions(psi):
    #psi[0] = 0
    #psi[-1] = 0
    return psi


L = 1
x = np.linspace(0, L, 50)
dx = x[1] - x[0]
V = 20*(x-L/2)**2

print(dt/dx**2)

# Declare potential: harmonic oscillator
V = 20*(x-L/2)**2

# Declare initial wavefunction
psi = np.sqrt(2/L)*np.sin(np.pi*x/L) + np.sqrt(2/L)*np.sin(2*np.pi*x/L)



# Initialize window
window = pyglet.window.Window()
gl.glClearColor(0, 0, 0, 1)

# Compile shader programs
vert_shader = Shader(vertex_source, 'vertex')
frag_shader = Shader(fragment_source, 'fragment')
program = ShaderProgram(vert_shader, frag_shader)
program.use()

# Set constant uniforms
#program.uniforms['color'].set(Vec3(255, 0, 0))
program.uniforms['translate'].set(Vec2(0.0, 0.0))

def createCurve(x, array, color):
    lines = np.zeros((len(x)*2-2, 2), dtype=float)
    pts = np.array([x, array]).T
    lines[::2] = pts[:-1]
    lines[1::2] = pts[1:]
    vlist = program.vertex_list(len(lines), gl.GL_LINES)
    vlist.position = lines.flatten()
    vlist.color = np.tile(np.array(color), len(lines))
    return lines, vlist

# Build vertex array for psi
psi_lines, psi_vlist = createCurve(x, np.conjugate(psi)*psi, (255, 0, 0))

# Build vertex array for V
V_lines, V_vlist = createCurve(x, V, (255, 255, 0))

@window.event
def on_draw():
    global psi, lines, x
    # Perform calculations
    psi = step(psi, H, (dx,), order=70)
    psi = boundaryConditions(psi)
    psi = normalize(psi, (dx,))
    # Push to vertex list
    pts = np.array([x, np.real(np.conjugate(psi)*psi)]).T
    psi_lines[::2] = pts[:-1]
    psi_lines[1::2] = pts[1:]
    psi_vlist.position = psi_lines.flatten()
    # Redraw lines
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    psi_vlist.draw(gl.GL_LINES)
    V_vlist.draw(gl.GL_LINES)


@window.event
def on_resize(width, height):
    gl.glViewport(0, 0, *window.get_framebuffer_size())
    program.uniforms['projection'].set(Mat4.orthogonal_projection(0, 1.0, 0, 5.0, -255, 255))
    

pyglet.app.run()
