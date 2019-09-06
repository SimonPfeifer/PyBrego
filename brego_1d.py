import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from riemannsolver import RiemannSolver
from sodshock_solution import get_solution

N_cells = 100
time = 0.2

space_order = 2
time_order = 1
boundary_type = 1 # periodic=0, reflective=1

global dt, GAMMA, BETA
dt = 0.001
dt_max = 0.1

GAMMA = 5. / 3
BETA = 0.3 # slope limiter factor

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

        self.ngb_left = None
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
        else:
            self.timestep = None
        if self.timestep is not None and self.timestep < dt:
            dt = self.timestep

    def pass2ngb(self):
        if self.ngb_left != None:
            self.ngb_left.density_ngb_right = self.density
            self.ngb_left.velocity_ngb_right = self.velocity
            self.ngb_left.pressure_ngb_right = self.pressure

    def slope_limiter(self, grad_phi_i, phi_i, phi_im1, phi_ip1):
        ''' Limits the gradient to ensure no new maxima or minima
            are created based on the neighbour cells. '''
        phi_max = max(phi_im1, phi_ip1)
        phi_min = min(phi_im1, phi_ip1)
        phi_im1_ext = phi_i - 0.5 * grad_phi_i
        phi_ip1_ext = phi_i + 0.5 * grad_phi_i
        phi_max_ext = max(phi_im1_ext, phi_ip1_ext)
        phi_min_ext = min(phi_im1_ext, phi_ip1_ext)
        if not phi_max_ext == phi_i:
            alpha_max = (phi_max - phi_i) / (phi_max_ext - phi_i)
        else:
            alpha_max = 1.
        if not phi_i == phi_min_ext:
            alpha_min = (phi_i - phi_min) / (phi_i - phi_min_ext)
        else:
            alpha_min = 1.
        return min(1., BETA * min(alpha_max, alpha_min)) * grad_phi_i

    def gradient(self):
        if self.boundary_type == 1 and self.boundary_x == 0:
            density_left = self.density
            velocity_left = -self.velocity
            pressure_left = self.pressure
        else:
            density_left = self.ngb_left.density
            velocity_left = self.ngb_left.velocity
            pressure_left = self.ngb_left.pressure

        if self.boundary_type == 1 and self.boundary_x == 1:
            density_right = self.density
            velocity_right = -self.velocity
            pressure_right = self.pressure
        else:
            density_right = self.density_ngb_right
            velocity_right = self.velocity_ngb_right
            pressure_right = self.pressure_ngb_right

        density_gradient = 0.5 * (density_right - density_left)
        velocity_gradient = 0.5 * (velocity_right - velocity_left)
        pressure_gradient = 0.5 * (pressure_right - pressure_left)

        self.density_gradient = self.slope_limiter(density_gradient, self.density, density_left, density_right)
        self.velocity_gradient = self.slope_limiter(velocity_gradient, self.velocity, velocity_left, velocity_right)
        self.pressure_gradient = self.slope_limiter(pressure_gradient, self.pressure, pressure_left, pressure_right)
      

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
    
    def update(self, space_order=1, time_order=1):
        ''' Solve the Reimann problem between self and neighbour cell using 
            primitive quantities. Then update the conserved quatities of self
            and neighbour cell. '''

        if self.boundary_type == 1 and self.boundary_x == 0:
            density_left = self.density
            velocity_left = -self.velocity
            pressure_left = self.pressure

            if space_order == 2:
                density_left += 0.5 * -self.density_gradient
                velocity_left += 0.5 * self.velocity_gradient
                pressure_left += 0.5 * -self.pressure_gradient

            if time_order == 2:
                density_left -= 0.5 * dt * (density_left * self.velocity_gradient +
                                                        velocity_left * -self.density_gradient)
                velocity_left -= 0.5 * dt * (velocity_left * self.velocity_gradient +
                                                        1. / density_left * -self.pressure_gradient)
                pressure_left -= 0.5 * dt * (GAMMA * pressure_left * self.velocity_gradient + 
                                                        velocity_left * -self.pressure_gradient)
        else:
            density_left = self.ngb_left.density
            velocity_left = self.ngb_left.velocity
            pressure_left = self.ngb_left.pressure

            if space_order == 2:
                density_left += 0.5 * self.ngb_left.density_gradient
                velocity_left += 0.5 * self.ngb_left.velocity_gradient
                pressure_left += 0.5 * self.ngb_left.pressure_gradient

            if time_order == 2:
                density_left -= 0.5 * dt * (density_left * self.ngb_left.velocity_gradient +
                                                        velocity_left * self.ngb_left.density_gradient)
                velocity_left -= 0.5 * dt * (velocity_left * self.ngb_left.velocity_gradient +
                                                        1. / density_left * self.ngb_left.pressure_gradient)
                pressure_left -= 0.5 * dt * (GAMMA * pressure_left * self.ngb_left.velocity_gradient + 
                                                        velocity_left * self.ngb_left.pressure_gradient)
        
        density_right = self.density
        velocity_right =  self.velocity
        pressure_right =  self.pressure

        if space_order == 2:
            density_right -= 0.5 * self.density_gradient
            velocity_right -= 0.5 * self.velocity_gradient
            pressure_right -= 0.5 * self.pressure_gradient

        if time_order == 2:
            density_right -= 0.5 * dt * (density_right * self.velocity_gradient +
                                                    velocity_right * self.density_gradient)
            velocity_right -= 0.5 * dt * (velocity_right * self.velocity_gradient +
                                                    1. / density_right * self.pressure_gradient)
            pressure_right -= 0.5 * dt * (GAMMA * pressure_right * self.velocity_gradient + 
                                                    velocity_right * self.pressure_gradient)

        flux_mass, flux_momentum, flux_energy = self.calculate_flux(density_left, velocity_left, 
                                                                                                    pressure_left, density_right, 
                                                                                                    velocity_right, pressure_right)

        if self.boundary_type != 1 or self.boundary_x == 1:
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

        
            if space_order == 2:
                density_left += 0.5 * self.density_gradient
                velocity_left += 0.5 * self.velocity_gradient
                pressure_left += 0.5 * self.pressure_gradient

                density_right -= 0.5 * -self.density_gradient
                velocity_right -= 0.5 * self.velocity_gradient
                pressure_right -= 0.5 * -self.pressure_gradient

            if time_order == 2:
                density_left -= 0.5 * dt * (density_left * self.velocity_gradient +
                                                        velocity_left * self.density_gradient)
                velocity_left -= 0.5 * dt * (velocity_left * self.velocity_gradient +
                                                        1. / density_left * self.pressure_gradient)
                pressure_left -= 0.5 * dt * (GAMMA * pressure_left * self.velocity_gradient + 
                                                        velocity_left * self.pressure_gradient)

                density_right -= 0.5 * dt * (density_right * self.velocity_gradient +
                                                        velocity_right * -self.density_gradient)
                velocity_right -= 0.5 * dt * (velocity_right * self.velocity_gradient +
                                                        1. / density_right * -self.pressure_gradient)
                pressure_right -= 0.5 * dt * (GAMMA * pressure_right * self.velocity_gradient + 
                                                        velocity_right * -self.pressure_gradient)

            flux_mass, flux_momentum, flux_energy = self.calculate_flux(density_left, velocity_left, 
                                                                                                        pressure_left, density_right, 
                                                                                                        velocity_right, pressure_right)

            self.mass -= flux_mass * self.surface_area * dt
            self.momentum -= flux_momentum * self.surface_area * dt
            self.energy -= flux_energy * self.surface_area * dt

    def check_conserved(self):
        ''' Perform checks on conserved quantities. '''
        assert self.mass >= 1E-20, 'ValueError: Mass is less than 0. {}'.format(self.mass)
        # assert self.momentum >= 1E-20, 'ValueError: Momentum is less than 0. {}'.format(self.momentum)
        assert self.energy >= 1E-20, 'ValueError: Energy is less than 0. {}'.format(self.energy)

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
    cell.volume = 1. / N_cells
    if i < N_cells/2.:
        cell.mass = 1. / N_cells
        cell.energy = 1. / (GAMMA - 1) / N_cells
    else:
        cell.mass = 0.125 / N_cells
        cell.energy = 0.1 / (GAMMA - 1) / N_cells
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
    for cell in cells:
        cell.check_conserved()
        cell.conserved2primitive()
        cell.pass2ngb()
        cell.calculate_timestep()
    if space_order == 2 or time_order == 2:
        for cell in cells:
            cell.gradient()
    for i, cell in enumerate(cells):
        cell.update(space_order=space_order, time_order=time_order)
    density.append([cell.density for cell in cells])

    time_current += dt
    pbar.update(dt)
    dt = copy.copy(dt_max)
pbar.close()

mass_start = np.sum([c.mass for c in cells])
momentum_start = np.sum([c.momentum for c in cells])
energy_start = np.sum([c.energy for c in cells])
print('End   ', mass_start, momentum_start, energy_start)   

fig = plt.figure()
ax = fig.add_subplot(111)
_, rho_sol, _, _ = get_solution(time)
ymax = np.nanmax(density)*1.1
for rho in density[::2]:
    ax.clear()
    ax.plot(np.arange(N_cells), rho)
    ax.plot(np.arange(1000)*N_cells/1000, rho_sol)
    ax.set_xlim([0,N_cells])
    ax.set_ylim([0,ymax])
    plt.pause(0.01)
plt.show()
