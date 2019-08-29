import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from riemannsolver import RiemannSolver

N_cells = 100
time = 2

boundary_type = 1 # periodic=0, reflective=1

global dt, GAMMA
dt_max = 0.1
dt = copy.copy(dt_max)
GAMMA = 5. / 3

class Cell:
    def __init__(self, boundary_type=0):
        self.volume = 0.
        self.mass = 0.
        self.momentum = 0.
        self.energy = 0.
        self.density = 0.
        self.velocity = 0.
        self.pressure = 0.

        self.surface_area = 1.
        self.timestep = None

        self.boundary_x = None
        self.boundary_type = boundary_type

    def conserved2primitive(self):
        ''' Convert conserved quantities to primitive quantities. '''
        self.density = self.mass / self.volume
        self.velocity = self.momentum / self.mass
        self.pressure = (GAMMA - 1) * (self.energy / self.volume - 0.5 * self.density * self.velocity**2)

    def calculate_timestep(self):
        ''' Calculate and update the safest minimum timestep. '''
        global dt
        if np.abs(self.velocity) > 0:
            self.timestep = self.volume / np.abs(self.velocity) * 0.2
        if self.timestep < dt:
            dt = self.timestep

    def calculate_flux(self, density_left, velocity_left, pressure_left, density_right, velocity_right, pressure_right):
        ''' Calculate the fluxes accross a boundary between 2 cells 
            for density, velcity and pressure using a Reimann solver. '''
        solver = RiemannSolver(GAMMA)
        density, velocity, pressure, _ = solver.solve(density_left, velocity_left, pressure_left,
                                                                            density_right, velocity_right, pressure_right)
        flux_mass = density * velocity
        flux_momentum = density * velocity**2 + pressure
        flux_energy = (pressure * GAMMA / (GAMMA - 1) + 0.5 * density * velocity**2) * velocity
        return flux_mass, flux_momentum, flux_energy
    
    def update(self):
        ''' Solve the Reimann problem between self and neighbour cell using 
            primitive quantities. Then update the conserved quatities of self
            and neighbour cell. '''

        if self.boundary_type == 1 and self.boundary_x == 0:
            density_left = self.density
            velocity_left = -self.velocity
            pressure_left = self.pressure

            density_right = self.density
            velocity_right =  self.velocity
            pressure_right =  self.pressure

            flux_mass, flux_momentum, flux_energy = self.calculate_flux(density_left, velocity_left, 
                                                                                                        pressure_left, density_right, 
                                                                                                        velocity_right, pressure_right)
            self.mass += flux_mass * self.surface_area * dt
            self.momentum += flux_momentum * self.surface_area * dt
            self.energy += flux_energy * self.surface_area * dt
        
        else:
            density_left = self.ngb_left.density
            velocity_left = self.ngb_left.velocity
            pressure_left = self.ngb_left.pressure

            density_right = self.density
            velocity_right =  self.velocity
            pressure_right =  self.pressure

            flux_mass, flux_momentum, flux_energy = self.calculate_flux(density_left, velocity_left, 
                                                                                                        pressure_left, density_right, 
                                                                                                        velocity_right, pressure_right)

            self.ngb_left.mass -= flux_mass * self.ngb_left.surface_area * dt
            self.ngb_left.momentum -= flux_momentum * self.ngb_left.surface_area * dt
            self.ngb_left.energy -= flux_energy  * self.ngb_left.surface_area * dt

            self.mass += flux_mass * self.surface_area * dt
            self.momentum += flux_momentum * self.surface_area * dt
            self.energy += flux_energy * self.surface_area * dt

            if self.boundary_type == 1 and self.boundary_x == 1:
                density_left = self.density
                velocity_left = self.velocity
                pressure_left = self.pressure

                density_right = self.density
                velocity_right =  -self.velocity
                pressure_right =  self.pressure

                flux_mass, flux_momentum, flux_energy = self.calculate_flux(density_left, velocity_left, 
                                                                                                            pressure_left, density_right, 
                                                                                                            velocity_right, pressure_right)
                self.mass -= flux_mass * self.surface_area * dt
                self.momentum -= flux_momentum * self.surface_area * dt
                self.energy -= flux_energy * self.surface_area * dt

    def check_conserved(self):
        ''' Perform checks on conserved quantities. '''
        assert self.mass >= -1E23, 'ValueError: Mass is less than 0. {}'.format(self.mass)
        assert self.momentum >= -1E23, 'ValueError: Momentum is less than 0. {}'.format(self.momentum)
        assert self.energy >= -1E23, 'ValueError: Energy is less than 0. {}'.format(self.energy)

    def copy_conserved(self, cell):
        ''' Copy primitive quantities from tagert cell. '''
        self.mass = copy.copy(cell.mass)
        self.momentum = copy.copy(cell.momentum)
        self.energy = copy.copy(cell.energy)
        self.volume = copy.copy(cell.volume)

# Initialise cells with initial conditions
cells = []
for i in range(N_cells):
    cell = Cell()
    cell.volume = 0.01
    cell.timestep = 0.01
    if i < N_cells/2.:
        cell.mass = 0.01
        cell.energy = 0.01 / (GAMMA - 1)
    else:
        cell.mass = 0.00125
        cell.energy = 0.001 / (GAMMA - 1)
    if i > 0:
        cell.ngb_left = cells[-1]

    # Set up boundary cells
    if i == 0:
        cell.boundary_x = 0
        cell.boundary_type = boundary_type
    if i == N_cells-1:
        cell.boundary_x = 1
        cell.boundary_type = boundary_type
    cells.append(cell)
if boundary_type == 0:
    cells[0].ngb_left = cells[-1]

mass_start = np.sum([c.mass for c in cells])
momentum_start = np.sum([c.momentum for c in cells])
energy_start = np.sum([c.energy for c in cells])
print('Start ', mass_start, momentum_start, energy_start)

density = []
time_current = 0.
pbar = tqdm(total=time, bar_format='{l_bar}{bar}| {n_fmt:.4s}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]')
while time_current < time: # Main timestep loop
    dt = copy.copy(dt_max)
    for cell in cells:
        cell.check_conserved()
        cell.conserved2primitive()
        cell.calculate_timestep()
    for i, cell in enumerate(cells):
        cell.update()
    density.append([cell.density for cell in cells])

    time_current +=dt
    pbar.update(dt)


mass_start = np.sum([c.mass for c in cells])
momentum_start = np.sum([c.momentum for c in cells])
energy_start = np.sum([c.energy for c in cells])
print('End   ', mass_start, momentum_start, energy_start)   

fig = plt.figure()
ax = fig.add_subplot(111)
ymax = np.nanmax(density)*1.1
for rho in density[::2]:
    ax.clear()
    ax.plot(np.arange(N_cells), rho)
    ax.set_xlim([0,N_cells])
    ax.set_ylim([0,ymax])
    plt.pause(0.01)
plt.show()