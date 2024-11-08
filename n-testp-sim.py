import rebound
import numpy as np
import os

# Parameters
theta = 30 * np.pi / 180  # Half of the total opening angle (in radians)
velocity = 1.0  # Unit velocity
n_periods = 10  # Number of periods to simulate
unit_length = 1.0  # Define the region of observation as unit length
output_folder = "sim_data"  # Folder to store the output files
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
    ps[1].ax += c  # Apply constant force in the x direction


# Function to run a single simulation
def run_simulation(simulation_number):
    sim = rebound.Simulation()
    sim.integrator = "ias15"  # integrator for test particles
    sim.dt = 0.001  # Small timestep

    # Add the massive stationary particle at the center
    sim.add(m=M)  # Central mass

    # Add the small particle with random initial conditions
    initial_distance = np.random.uniform(0.0, unit_length)
    theta = np.arccos(np.random.uniform(-1, 1))  # Polar angle
    phi = np.random.uniform(0, np.pi)  # Azimuthal angle
    x = initial_distance * np.sin(phi) * np.cos(theta)
    y = initial_distance * np.sin(phi) * np.sin(theta)
    z = initial_distance * np.cos(phi)


    velocity_angle_theta = np.random.uniform(0, 2 * np.pi)  # Random velocity direction
    velocity_angle_phi = np.random.uniform(0, np.pi)
    vx = velocity * np.sin(velocity_angle_phi) * np.cos(velocity_angle_theta)
    vy = velocity * np.sin(velocity_angle_phi) * np.sin(velocity_angle_theta)
    vz = velocity * np.cos(velocity_angle_phi)

    # Add the small particle as a test particle (mass=0)
    sim.add(x=x, y=y,z=z, vx=vx, vy=vy, vz=vz)  # Mass is 0 by default for test particles

    # Set N_active to ensure only the central mass influences the test particles
    sim.N_active = 1  # Only the central mass is active

    # Calculate the orbital period based on initial distance
    orbital_period = calculate_orbital_period(initial_distance, M)
    total_time = n_periods * orbital_period  # Total time to simulate

    # Add the additional force to the simulation
    sim.additional_forces = 0
    
    left_time_count = 0
    right_time_count = 0

    # Define the output file for this simulation
    output_file = os.path.join(output_folder, f"simulation_{simulation_number}.txt")
    
    with open(output_file, "w") as f:
        f.write("Time step, x, y, vx, vy\n")
        while sim.t < total_time:
            sim.integrate(sim.t + sim.dt)  # Integrate simulation
            x, y, z = sim.particles[1].x, sim.particles[1].y, sim.particles[1].z
            vx, vy, vz = sim.particles[1].vx, sim.particles[1].vy, sim.particles[1].vz

            # Record the position and velocity every 10 time steps
            f.write(f"{sim.t:.6f}, {x:.6f}, {y:.6f}, {z:.6f}, {vx:.6f}, {vy:.6f}, {vz:.6f}\n")

# Running the simulation 100 times with test particles
n_simulations = 10

for simulation_number in range(1, n_simulations + 1):
    run_simulation(simulation_number)

print("Simulations completed!")