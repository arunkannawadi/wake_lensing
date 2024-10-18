import rebound
import numpy as np
import os
import time  # Import the time module

# Parameters
theta = 30 * np.pi / 180  # Half of the total opening angle (in radians)
velocity = 1.0  # Unit velocity
n_periods = 2  # Number of periods to simulate
unit_length = 1.0  # Define the region of observation as unit length
output_folder = "simd"  # Folder to store the output files
np.random.seed(42)

# Gravitational constant and central mass
G = 1.0
M = 1.0  # Central mass
c = 0.01  # Constant force magnitude

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to calculate orbital period based on initial distance
def calculate_orbital_period(distance, M):
    return 2 * np.pi * np.sqrt(distance**3 / (G * M))

# Define the constant force (Stark force)
def constant_force(reb_sim):
    ps = reb_sim.contents.particles  # Access particles via reb_sim.contents
    for i in range(1, reb_sim.contents.N):
        ps[i].ax += 0  # Apply constant force in the x direction for each test particle

# Function to run a simulation with multiple test particles
def run_simulation_with_particles(n_particles):
    sim = rebound.Simulation()
    sim.integrator = "ias15"  # Fast integrator for test particles
    sim.dt = 0.01  # Small timestep

    # Add the massive stationary particle at the center
    sim.add(m=M)  # Central mass

    # Add multiple small particles with random initial conditions (using polar coordinates)
    orbital_periods = []
    for i in range(n_particles):
        initial_distance = np.random.uniform(0.0, unit_length)
        theta = np.random.uniform(0, 2 * np.pi)  # Polar angle
        phi = np.random.uniform(0, np.pi)  # Azimuthal angle
        x = initial_distance * np.sin(phi) * np.cos(theta)
        y = initial_distance * np.sin(phi) * np.sin(theta)
        z = initial_distance * np.cos(phi)


        velocity_angle_theta = np.random.uniform(0, 2 * np.pi)  # Random velocity direction
        velocity_angle_phi = np.random.uniform(0, np.pi)
        vx = velocity * np.sin(velocity_angle_phi) * np.cos(velocity_angle_theta)
        vy = velocity * np.sin(velocity_angle_phi) * np.sin(velocity_angle_theta)
        vz = velocity * np.cos(velocity_angle_phi)

        
        sim.add(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)  # Mass is 0 by default for test particles

        # Calculate the orbital period for this particle
        orbital_period = calculate_orbital_period(initial_distance, M)
        orbital_periods.append(orbital_period)

    # Set N_active to ensure only the central mass influences the test particles
    sim.N_active = 1  # Only the central mass is active

    # Add the additional force to the simulation
    sim.additional_forces = constant_force

    # Define the total simulation time based on the longest orbital period
    total_time = n_periods * max(orbital_periods)  # Total time to simulate

    # Define the output file for this simulation
    output_file = os.path.join(output_folder, f"combined_simulation.txt")

    with open(output_file, "w") as f:
        f.write("Particle, Time step, r, theta\n")
        buffer = []
        while sim.t < total_time:
            sim.integrate(sim.t + sim.dt)  # Integrate simulation
            for i in range(1, sim.N):
                buffer.append(f"{i}, {sim.t:.6f}, {sim.particles[i].x:.6f}, {sim.particles[i].y:.6f}, {sim.particles[i].z:.6f}\n")
            if len(buffer) >= 1000:  # Write in chunks
                f.writelines(buffer)
                buffer = []
        if buffer:  # Write remaining data
            f.writelines(buffer)


# Start the timer
start_time = time.time()

# Running the simulation with 100 particles
n_particles = 10
run_simulation_with_particles(n_particles)

# End the timer
end_time = time.time()

# Calculate runtime
runtime = end_time - start_time
print(f"Simulation with {n_particles} particles completed in {runtime:.2f} seconds.")

