import copy
import numpy as np
import matplotlib.pyplot as plt
from riemannsolver import RiemannSolver

N_cells = 100
N_timestep = 1000

boundary_reflective = True

global dt, GAMMA
dt = 0.001
GAMMA = 5. / 3

class Cell:
    def __init__(self, reflective=False):
        self.volume = 0.
        self.mass = 0.
        self.momentum = 0.
        self.energy = 0.
        self.density = 0.
        self.velocity = 0.
        self.pressure = 0.

        self.ngb_left = None
        self.surface_area = 1.

        self.reflective = reflective

    def conserved2primitive(self):
        ''' Convert conserved quantities to primitive quantities. '''
        self.density = self.mass / self.volume
        self.velocity = self.momentum / self.mass
        self.pressure = (GAMMA - 1) * (self.energy / self.volume - 0.5 * self.density * self.velocity**2)

    def update(self):
        ''' Solve the Reimann problem between self and neighbour cell using 
            primitive quantities. Then update the conserved quatities of self
            and neighbour cell. '''
        solver = RiemannSolver(GAMMA)
        density, velocity, pressure, _ = solver.solve(self.ngb_left.density, self.ngb_left.velocity, self.ngb_left.pressure,
                                                                                self.density, self.velocity, self.pressure)

        flux_mass = density * velocity
        flux_momentum = density * velocity**2 + pressure
        flux_energy = (pressure * GAMMA / (GAMMA - 1) + 0.5 * density * velocity**2) * velocity

        self.ngb_left.mass -= flux_mass * self.ngb_left.surface_area * dt
        self.ngb_left.momentum -= flux_momentum * self.ngb_left.surface_area * dt
        self.ngb_left.energy -= flux_energy  * self.ngb_left.surface_area * dt

        self.mass += flux_mass * self.surface_area * dt
        self.momentum += flux_momentum * self.surface_area * dt
        self.energy += flux_energy * self.surface_area * dt

    def copy_from_cell(self, cell):
        ''' Copy primitive quantities from tagert cell. '''
        self.density = copy.copy(cell.density)
        self.velocity = copy.copy(cell.velocity)
        self.pressure = copy.copy(cell.pressure)
        if self.reflective:
            self.velocity *= -1.

# Initialise cells with initial conditions
cells = []
for i in range(N_cells):
    cell = Cell()
    cell.volume = 0.01
    
    if i > N_cells*2./4. and i < N_cells*3./4.:
        cell.mass = 0.01
        cell.energy = 0.01 / (GAMMA - 1)
    else:
        cell.mass = 0.00125
        cell.energy = 0.001 / (GAMMA - 1)

    if i > 0:
        cell.ngb_left = cells[-1]
    cells.append(cell)

# Add boundary cells
cells_boundary = []
for i in range(2):
    cells_boundary.append(Cell(reflective=boundary_reflective))
cells[0].ngb_left = cells_boundary[0]
cells_boundary[-1].ngb_left = cells[-1]

density = np.zeros([N_timestep, N_cells])
for j in range(N_timestep): # Main timestep loop
    for cell in cells:
        cell.conserved2primitive()
    for i, cell in enumerate(cells_boundary):
        sign = (i < len(cells_boundary)/2.)
        sign = sign * 2 - 1
        cell.copy_from_cell(cells[sign*2])
        if sign < 0:
            cell.update()
    for cell in cells:
        cell.update()

    density[j] = [cell.density for cell in cells]

fig = plt.figure()
ax = fig.add_subplot(111)
for rho in density[::2]:
    ax.clear()
    ax.plot(np.arange(N_cells), rho)
    ax.set_ylim([0,2])
    plt.pause(0.0001)
plt.show()