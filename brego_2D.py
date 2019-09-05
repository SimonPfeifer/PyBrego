import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from riemannsolver import RiemannSolver
from sodshock_solution import get_solution

Nx_cells = 20
Ny_cells = 20
time = 1.

boundary_type = 0 # periodic=0, reflective=1

global dt, GAMMA
dt = 0.001
dt_max = 0.1

GAMMA = 5. / 3

class Cell:
    def __init__(self, boundary_type=0):
        self.midpoint = np.array([0., 0.])
        self.mass = 0.
        self.momentum = np.array([0., 0.])
        self.energy = 0.
        self.density = 0.
        self.velocity = np.array([0., 0.])
        self.pressure = 0.

        self.volume = 0.
        self.surface_area_x = 1.
        self.surface_area_y = 1.

        self.ngb_left = None
        self.ngb_up = None
        self.boundary_x = None
        self.boundary_y = None
        self.boundary_type = boundary_type

    def calculate_timestep(self):
        ''' Calculate and update the safest minimum timestep. '''
        global dt
        mask = np.abs(self.velocity) > 1E-20
        if np.any(mask):
            timestep = np.min(self.volume / np.abs(self.velocity)[mask]) * 0.2
            if timestep < dt:
                dt = timestep

    def check_conserved(self):
        ''' Perform checks on conserved quantities. '''
        assert self.mass >= 1E-20, 'ValueError: Mass is less than 0. {}'.format(self.mass)
        assert self.energy >= 1E-20, 'ValueError: Energy is less than 0. {}'.format(self.energy)

    def conserved2primitive(self):
        ''' Convert conserved quantities to primitive quantities. '''
        self.density = self.mass / self.volume
        self.velocity = self.momentum / self.mass
        self.pressure = (GAMMA - 1) * (self.energy / self.volume - 0.5 * self.density * np.sum(self.velocity**2))

    def calculate_flux(self, density_left, velocity_left, pressure_left, density_right, velocity_right, pressure_right, index):
        ''' Calculate the fluxes accross a boundary between 2 cells 
            for density, velcity and pressure using a Reimann solver. '''
        solver = RiemannSolver(GAMMA)
        density, velocity, pressure, flag = solver.solve(density_left, velocity_left[index], pressure_left,
                                                                            density_right, velocity_right[index], pressure_right)
        if flag == -1:
            velocity_perp = velocity_left[idx2]
        elif flag == 1:
            velocity_perp = velocity_right[idx2]
        else:
            raise(ValueError, 'Riemann solver output flag must be either -1 or 1: {}'.format(flag))

    def update(self):


        for i in range(2):
            if i == 0:
                idx1 = 0
                idx2 = 1
            elif i == 1:
                idx1 = 1
                idx2 = 0

            if i == 0:
                density_left = self.ngb_left.density
                velocity_left = self.ngb_left.velocity
                pressure_left = self.ngb_left.pressure
            elif i == 1:
                density_left = self.ngb_up.density
                velocity_left = self.ngb_up.velocity
                pressure_left = self.ngb_up.pressure

            density_right = self.density
            velocity_right = self.velocity
            pressure_right = self.pressure

            solver = RiemannSolver(GAMMA)
            density, velocity, pressure, flag = solver.solve(density_left, velocity_left[idx1], pressure_left,
                                                                                density_right, velocity_right[idx1], pressure_right)
            if flag == -1:
                velocity_perp = velocity_left[idx2]
            elif flag == 1:
                velocity_perp = velocity_right[idx2]
            else:
                raise(ValueError, 'Riemann solver output flag must be either -1 or 1: {}'.format(flag))

            flux_mass = density * velocity
            flux_momentum = density * velocity**2 + pressure
            flux_momentum_perp = density * velocity * velocity_perp
            flux_energy = (pressure * GAMMA / (GAMMA - 1) + 0.5 * density * velocity**2) * velocity

            if i == 0:
                self.ngb_left.mass -= flux_mass * self.ngb_left.surface_area_x * dt
                self.ngb_left.momentum[idx1] -= flux_momentum * self.ngb_left.surface_area_x * dt
                self.ngb_left.momentum[idx2] -= flux_momentum_perp * self.ngb_left.surface_area_x * dt
                self.ngb_left.energy -= flux_energy  * self.ngb_left.surface_area_x * dt
            elif i == 1:
                self.ngb_up.mass -= flux_mass * self.ngb_up.surface_area_y * dt
                self.ngb_up.momentum[1] -= flux_momentum * self.ngb_up.surface_area_y * dt
                self.ngb_up.momentum[0] -= flux_momentum_perp * self.ngb_up.surface_area_y * dt
                self.ngb_up.energy -= flux_energy  * self.ngb_up.surface_area_y * dt

            self.mass += flux_mass * self.surface_area_x * dt
            self.momentum[idx1] += flux_momentum * self.surface_area_x * dt
            self.momentum[idx2] += flux_momentum_perp * self.surface_area_x * dt
            self.energy += flux_energy * self.surface_area_x * dt

# Initialise cells with initial conditions
cells = []
for i in range(Ny_cells):
    for j in range(Nx_cells):
        cell = Cell()
        cell.midpoint[0] = (j + 0.5) / Nx_cells
        cell.midpoint[1] = (i + 0.5) / Ny_cells
        cell.volume = 1. / Nx_cells / Ny_cells
        cell.surface_area_x = 1. / Nx_cells
        cell.surface_area_y = 1. / Ny_cells
        if j < Nx_cells/2. and i < Ny_cells/2.:
            cell.mass = 1. / Nx_cells / Ny_cells
            cell.energy = 1. / (GAMMA - 1) / Nx_cells / Ny_cells
        else:
            cell.mass = 0.125 / Nx_cells / Ny_cells
            cell.energy = 0.1 / (GAMMA - 1) / Nx_cells / Ny_cells
        if j > 0:
            cell.ngb_left = cells[i * Nx_cells + j - 1]
        if i > 0:
            cell.ngb_up = cells[(i - 1) * Nx_cells + j]

        # Set up boundary cells
        if j == 0:
            cell.boundary_x = 0
            cell.boundary_type = boundary_type
        if i == 0:
            cell.boundary_y = 0
            cell.boundary_type = boundary_type
        if j == Nx_cells-1:
            cell.boundary_x = 1
            cell.boundary_type = boundary_type
        if i == Ny_cells-1:
            cell.boundary_y = 1
            cell.boundary_type = boundary_type
        cells.append(cell)

if boundary_type == 0:
    for i in range(Ny_cells):
        cells[i * Nx_cells + 0].ngb_left = cells[i * Nx_cells + Nx_cells - 1]
    for j in range(Nx_cells):
        cells[0 * Nx_cells + j].ngb_up = cells[(Ny_cells - 1) * Nx_cells + j]

mass_start = np.sum([cell.mass for cell in cells])
momentum_start = np.sum([cell.momentum for cell in cells])
energy_start = np.sum([cell.energy for cell in cells])
print('START')
print('Mass: ', mass_start)
print('Momentum: ', momentum_start)
print('Energy: ', energy_start)

n_iter = 0
time_current = 0.
density = np.zeros([int(time/dt*100), Nx_cells*Ny_cells])
pbar = tqdm(total=time, bar_format='{l_bar}{bar}| {n_fmt:.4s}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]')
while time_current < time:
    for cell in cells:
        cell.check_conserved()
        cell.conserved2primitive()
        cell.calculate_timestep()
    for cell in cells:
        cell.update()

    density[n_iter] = [cell.density for cell in cells]
    n_iter += 1
    time_current += dt
    pbar.update(dt)
    dt = copy.copy(dt_max)
pbar.close()

mass_start = np.sum([cell.mass for cell in cells])
momentum_start = np.sum([cell.momentum for cell in cells])
energy_start = np.sum([cell.energy for cell in cells])
print('END')
print('Mass: ', mass_start)
print('Momentum: ', momentum_start)
print('Energy: ', energy_start)

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(Nx_cells)/Nx_cells+0.5, np.arange(Ny_cells)/Ny_cells+0.5) 
_, rho_sol, _, _ = get_solution(time)
vmax = np.nanmax(density)*1.1
for rho in density[::5]:
    ax.clear()
    ax.plot_wireframe(X, Y, rho.reshape(Ny_cells, Nx_cells))
    ax.set_zlim([0, vmax])
    plt.pause(0.01)
plt.show()
