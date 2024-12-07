import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rebound
import os

# Visualization parameters
n_particles = 10  # Number of particles to visualize
theta = 30 * np.pi / 180  # Half of the total opening angle (in radians)
velocity = 1.0  # Unit velocity
n_periods = 2  # Number of periods to simulate
unit_length = 1.0  # Define the region of observation as unit length
output_folder = "simd"  # Folder to store the output files

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Gravitational constant and central mass
G = 1.0
M = 1.0  # Central mass
c = 0.01  # Constant force magnitude

# Function to calculate orbital period based on initial distance
def calculate_orbital_period(distance, M):
    return 2 * np.pi * np.sqrt(distance**3 / (G * M))

# Define the constant force (Stark force)
def constant_force(reb_sim):
    ps = reb_sim.contents.particles
    for i in range(1, reb_sim.contents.N):
        ps[i].ax += c  # Apply constant force in the x direction

# Function to run a simulation and return positions
def run_simulation_with_particles(n_particles):
    sim = rebound.Simulation()
    sim.integrator = "whfast"  # Fast integrator
    sim.dt = 0.001  # Small timestep

    # Add the central mass particle
    sim.add(m=M)  # Central mass

    positions = []

    orbital_periods = []
    for i in range(n_particles):
        initial_distance = np.random.uniform(0.0, unit_length)
        initial_angle = np.random.uniform(0, 2 * np.pi)
        
        sim.add(a=initial_distance, e=0, inc=0, Omega=0, omega=0, f=initial_angle)
        orbital_period = calculate_orbital_period(initial_distance, M)
        orbital_periods.append(orbital_period)

    sim.N_active = 1  # Only central mass affects the particles
    sim.additional_forces = constant_force  # Apply external force

    total_time = n_periods * max(orbital_periods)
    time_steps = []

    # Run the simulation and record positions
    while sim.t < total_time:
        sim.integrate(sim.t + sim.dt)
        step_positions = [(p.x, p.y) for p in sim.particles[1:]]  # Exclude central mass
        positions.append(step_positions)
        time_steps.append(sim.t)

    return positions, time_steps

# Run the simulation
positions, time_steps = run_simulation_with_particles(n_particles)

# Convert Cartesian coordinates to polar
def cartesian_to_polar(positions):
    polar_positions = []
    for step in positions:
        polar_step = []
        for (x, y) in step:
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            polar_step.append((r, phi))
        polar_positions.append(polar_step)
    return polar_positions

polar_positions = cartesian_to_polar(positions)

# Set up the plot for visualization
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_ylim(0, unit_length)

# Initialize particles' positions
particles, = ax.plot([], [], 'bo', markersize=5)

# Function to initialize the plot
def init():
    particles.set_data([], [])
    return particles,

# Function to update the plot at each frame
def update(frame):
    r_values = [pos[0] for pos in polar_positions[frame]]
    theta_values = [pos[1] for pos in polar_positions[frame]]
    particles.set_data(theta_values, r_values)
    return particles,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(time_steps), init_func=init, blit=True, interval=20)

# Show the animation
plt.show()
