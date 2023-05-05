"""
This module contains classes for rendering the results of a simulation onto a live viewer
"""
# Import modules
import numpy as np
from scipy.spatial import Delaunay
from .Simulator import Simulation
from .Colormaps import viridis
from typing import Callable, Union, Tuple
from zlib import decompress
from base64 import b64decode
from io import BytesIO
from dataclasses import dataclass
from time import time
from enum import Enum

# Import pyglet stuff
import pyglet
from pyglet.window import key
import pyglet.gl as gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec2

__all__ = ["GLRender1D", "GLRender2D", "RenderMode"]

# Apply windows-specific fix
from sys import platform
if platform.startswith('win'):
    import ctypes
    appid = u'qsim.renderer.thing'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)


class RenderMode(Enum):
    SQUARE_MODULUS = "Square Modulus"
    REAL_PART = "Real Part"
    IMAGINARY_PART = "Imaginary Part"


@dataclass
class Extent:
    xlo: float = 0.0
    xhi: float = 1.0
    ylo: float = 0.0
    yhi: float = 5.0


class GLRenderer(pyglet.window.Window):
    """
    Base class for OpenGL-based renderers
    """
    __icon = b'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'

    def __init__(self, *args, resizable=True, **kwargs):
        super().__init__(*args, resizable=resizable, **kwargs)

        # Configure window
        self.set_caption("Quantum Simulator")
        buffer = BytesIO(decompress(b64decode(self.__icon)))
        image = pyglet.image.load('icon.png', buffer)
        self.set_icon(image)

        # Configure GL stuff
        gl.glClearColor(0, 0, 0, 1)
        self.batch = pyglet.graphics.Batch()
        self.curves = []

        # Compile shader programs
        vert_shader = Shader(self.getVertexShaderSource(), 'vertex')
        frag_shader = Shader(self.getFragmentShaderSource(), 'fragment')
        self.program = ShaderProgram(vert_shader, frag_shader)
        self.program.uniforms['translate'].set(Vec2(0.0, 0.0))
        self.program.use()

        # Configure viewport extent
        self.extent = Extent(0.0, 1.0, 0.0, 5.0)
        self._updateProjectionMatrix()

        # Simulation reference
        self.simulation: Union[None, Simulation] = None

        # Create info labels
        self.energy_label = pyglet.text.Label("Energy: no", font_size=16, x=6, y=self.height, anchor_y='top')
        self.fps_label = pyglet.text.Label("FPS: no", font_size=16, x=10, y=self.height - 36, anchor_y='top')

        # Other parameters
        self.show_fps = False
        self.last_time: float = time()
        self.show_energy = False
        self.paused = False
        self.render_mode: RenderMode = RenderMode.SQUARE_MODULUS

        self.initial_height = self.height
        self.initial_width = self.width

    def applyVertexArray(self, simulation: Simulation):
        """Pushes the simulation's current wavefunction into a vertex array.  Override in subclass"""
        raise NotImplementedError()

    def on_draw(self):
        # Apply simulation step
        if not self.paused:
            self.simulation.step()

        self.applyVertexArray(self.simulation)

        # Clear screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Draw everything in the batch
        self.batch.draw()

        # Draw label text
        if self.show_fps:
            self.fps_label.text = f'FPS: {1.0/(time() - self.last_time):.0f}'
            self.fps_label.draw()
        if self.show_energy:
            self.energy_label.text = f'〈E〉: {self.simulation.energy():.2f}'
            self.energy_label.draw()

        # Update current time
        self.last_time = time()

    def on_key_press(self, symbol, modifiers):
        # F: Toggle FPS indicator
        if symbol == key.F:
            self.show_fps = not self.show_fps

        # E: Toggle energy indicator
        elif symbol == key.E:
            self.show_energy = not self.show_energy

        # Space: Toggle pause:
        elif symbol == key.SPACE:
            self.paused = not self.paused

        # Left arrow: advance forward ONE step
        elif symbol == key.RIGHT:
            self.simulation.step()

    def setRenderMode(self, mode: RenderMode):
        """
        Configure what part of the wavefunction to render.
        :param mode: Render mode.  Available options are: 'Square Modulus', 'Real Part', and 'Imaginary Part'
        """
        # Enforce type
        mode = RenderMode(mode)

        # Set mode
        self.render_mode = mode

    def setExtent(self, xlo: float = None, xhi: float = None, ylo: float = None, yhi: float = None):
        """
        Sets the extent of the viewport

        :param xlo: X-coordinate of left-end of viewport
        :param xhi: X-coordinate of right-end of viewport
        :param ylo: Y-coordinate of bottom-end of viewport
        :param yhi: Y-coordinate of top-end of viewport
        """
        # Optionally update each bound
        if xlo is not None:
            self.extent.xlo = float(xlo)
        if xhi is not None:
            self.extent.xhi = float(xhi)
        if ylo is not None:
            self.extent.ylo = float(ylo)
        if yhi is not None:
            self.extent.yhi = float(yhi)

        # Update the projection matrix
        self._updateProjectionMatrix()

    def _updateProjectionMatrix(self):
        """Internal helper function that reloads the projection matrix based on the current viewport settings"""
        self.program.uniforms['projection'].set(Mat4.orthogonal_projection(self.extent.xlo, self.extent.xhi,
                                                                           self.extent.ylo, self.extent.yhi,
                                                                           -255, 255))

    def on_resize(self, width, height):
        gl.glViewport(0, 0, *self.get_framebuffer_size())

    def attachSimulation(self, simulation: Simulation):
        """
        Attaches the simulator object to the window's main loop
        """
        # Store reference to simulation object
        self.simulation = simulation

        # Set x-bounds based on simulation's internal meshgrid
        x = simulation.meshgrid[0]
        self.setExtent(xlo=np.min(x), xhi=np.max(x))

    def setOffset(self, x: float, y: float):
        self.program.uniforms['translate'].set(Vec2(float(x), float(y)))

    @classmethod
    def getVertexShaderSource(cls):
        return """#version 150 core
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

    @classmethod
    def getFragmentShaderSource(cls):
        return """#version 150 core
                  in vec4 vertex_color;
                  in vec2 fragCoord;
                  out vec4 final_color;

                  void main()
                  {
                      final_color = vertex_color;
                  }
                  """

    @classmethod
    def start(cls):
        pyglet.app.run()


class GLRender1D(GLRenderer):
    """
    Renderer specifically for 1D wavefunctions
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Vertex arrays
        self.vlist = None
        self._vlist_length_old = -1

        # Configuration options
        self.color = (255, 0, 0)

    def setStateColor(self, color: tuple):
        self.color = tuple(color)[:3]

    def attachSimulation(self, simulation: Simulation):
        """
        Attach the simulator to the renderer.  Upon each draw call, the simulation will be advanced by one step.

        :param simulation: Simulator object
        """
        # Check state dimension
        if simulation.dims != 1:
            raise ValueError(f"{simulation.dims}-dimensional system not compatible with a 1D viewer")

        # Call super-class function
        super().attachSimulation(simulation)

        # Build vertex list for the state vector
        self.applyVertexArray(simulation)

    @staticmethod
    def _arraysToLines(x: np.ndarray, y: np.ndarray):
        """Helper function that converts two arrays of x/y values into an array of line segments"""
        lines = np.zeros((len(x) * 2 - 2, 2), dtype=float)
        pts = np.array([x, y]).T
        lines[::2] = pts[:-1]
        lines[1::2] = pts[1:]
        return lines

    def applyRenderMode(self, psi: np.ndarray) -> np.ndarray:
        """Processes the provided array based on the current rendering mode"""
        if self.render_mode == RenderMode.SQUARE_MODULUS:
            return np.real(np.conjugate(psi)*psi)

        elif self.render_mode == RenderMode.REAL_PART:
            return np.real(psi)

        elif self.render_mode == RenderMode.IMAGINARY_PART:
            return np.imag(psi)

    def applyVertexArray(self, simulation: Simulation):
        x = simulation.meshgrid[0]
        psi = self.applyRenderMode(simulation.psi)
        lines = self._arraysToLines(x, psi)
        if len(lines) != self._vlist_length_old:
            self._vlist_length_old = len(lines)
            self.vlist = self.program.vertex_list(len(lines), gl.GL_LINES, batch=self.batch)
        self.vlist.position = lines.flatten()
        self.vlist.color = np.tile(np.array(self.color), len(lines))

    def addCurve(self, func: Callable, color: Union[Tuple[int, int, int], list, np.ndarray]):
        """
        Adds a curve to the plot

        :param func: Function used to generate line
        :param color: A three-tuple of integers ranging 0-255
        """
        # Raise exception if there's no coordinate system to evaluate line against
        if self.simulation is None:
            raise ValueError("Unable to add line before simulation has been declared!")

        # Enforce color formatting
        color = np.array(color, dtype=int)
        if np.ndim(color) > 1:
            raise ValueError(f"Expected 'color' to be a 1D array, was {np.ndim(color)}-D")
        if len(color) < 3:
            raise ValueError(f"Expected 'color' to be a 1D array of length 3, was {len(color)}")
        color = color[:3]

        # Evaluate function
        x = self.simulation.meshgrid[0]
        y = func(x)

        # Generate lines and add to window
        lines = self._arraysToLines(x, y)
        vlist = self.program.vertex_list(len(lines), gl.GL_LINES, batch=self.batch)

        vlist.position = lines.flatten()
        vlist.color = np.tile(color, len(lines))

        # Add to list to prevent garbage collection
        self.curves.append(vlist)

    def clearCurves(self):
        """Removes any additional curves from the plot"""
        for vlist in self.curves:
            vlist.delete()
        self.curves = []


class GLRender2D(GLRenderer):
    """
    Renderer specifically for 2D wavefunctions
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Vertex arrays
        self.vlist = None
        self.boxes = []

        # Place to store triangle mesh
        self._mesh: np.ndarray = np.array([], dtype=float)
        self._simplices: np.ndarray = np.array([], dtype=int)

        # Configuration options
        self.colormap: Callable = viridis
        self.vmin: float = 0.0
        self.vmax: float = 2.0

    def attachSimulation(self, simulation: Simulation):
        """
        Attach the simulator to the renderer.  Upon each draw call, the simulation will be advanced by one step.

        :param simulation: Simulator object
        """
        # Check state dimension
        if simulation.dims != 2:
            raise ValueError(f"{simulation.dims}-dimensional system not compatible with a 2D viewer")

        # Call super-class function
        super().attachSimulation(simulation)

        # Build vertex list for the state vector
        self.rebuildMesh(simulation)

        # Set x-bounds based on simulation's internal meshgrid
        x = simulation.meshgrid[0]
        y = simulation.meshgrid[1]
        self.setExtent(xlo=np.min(x), xhi=np.max(x),
                       ylo=np.min(y), yhi=np.max(y))

    def rebuildMesh(self, simulation: Simulation):
        """Helper function that builds a triangle mesh to cover the simulation's entire coordinate system"""
        # Extract axes
        axes = simulation.meshgrid
        pts = np.array([axes[0], axes[1]]).T.reshape(-1, 2)

        # Convert to triangle mesh
        tri = Delaunay(pts)
        self._simplices = tri.simplices
        self._mesh = pts[tri.simplices]

        # Create vertex array object and assign mesh
        self.vlist = self.program.vertex_list(len(self._mesh)*3, gl.GL_TRIANGLES, batch=self.batch)
        self.vlist.position = self._mesh.flatten()

        # Assign initial colors
        self.applyVertexArray(simulation)

    def setColorRange(self, vmin: float = None, vmax: float = None):
        """
        Sets the colormap range.
        :param vmin: Lower bound of the colormap
        :param vmax: Upper bound of the colormap
        """
        if vmin is not None:
            self.vmin = float(vmin)
        if vmax is not None:
            self.vmax = float(vmax)

    def applyVertexArray(self, simulation: Simulation):
        """
        Pushes data from the provided simulation object to the viewport
        :param simulation: Simulation object to pull data from
        """
        psi2 = simulation.squareMod
        colors = viridis(psi2, self.vmin, self.vmax).reshape(-1, 3)
        self.vlist.color = colors[self._simplices].flatten()

    def addBox(self, x1: float, y1: float, x2: float, y2: float, color: Tuple[int, int, int]):
        """
        Adds a box to the plot

        :param x1: x₁ coordinate of box
        :param y1: y₁ coordinate of box
        :param x2: x₂ coordinate of box
        :param y2: y₂ coordinate of box
        :param color: A three-tuple of integers ranging 0-255
        """
        # Raise exception if there's no coordinate system to evaluate line against
        if self.simulation is None:
            raise ValueError("Unable to add line before simulation has been declared!")

        # Enforce color formatting
        color = np.array(color, dtype=int)
        if np.ndim(color) > 1:
            raise ValueError(f"Expected 'color' to be a 1D array, was {np.ndim(color)}-D")
        if len(color) < 3:
            raise ValueError(f"Expected 'color' to be a 1D array of length 3, was {len(color)}")
        color = color[:3]

        # Convert box to triangle mesh
        pts = np.array([[x1, y1],
                        [x1, y2],
                        [x2, y2],
                        [x2, y1]], dtype=float)
        tri = Delaunay(pts)
        mesh = pts[tri.simplices]

        # Generate vertex array
        vlist = self.program.vertex_list(len(mesh)*3, gl.GL_TRIANGLES, batch=self.batch)
        vlist.position = mesh.flatten()
        vlist.color = np.tile(color, mesh.size//2)

        # Add to list to prevent garbage collection
        self.boxes.append(vlist)

    def clearBoxes(self):
        """Clears all boxed from the canvas"""
        for vlist in self.boxes:
            vlist.delete()
        self.boxes = []
