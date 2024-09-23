import rebound
import numpy as np
import os

# Parameters
theta = 30 * np.pi / 180  # Half of the total opening angle (in radians)
velocity = 1.0  # Unit velocity
n_periods = 10  # Number of periods to simulate
unit_length = 1.0  # Define the region of observation as unit length
output_folder = "simulation_data_updated"  # Folder to store the output files
np.random.seed(42)

# Gravitational constant and central mass
G = 1.0
M = 1.0  # Central mass
c = 0.01  # Constant force magnitude

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to check if a point is inside a triangular region
def in_triangle(x, y, left=True):
    if left:
        return x <= 0 and np.abs(y) <= -x * np.tan(theta)
    else:
        return x >= 0 and np.abs(y) <= x * np.tan(theta)

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
    sim.integrator = "whfast"  # Fast integrator for test particles
    sim.dt = 0.001  # Small timestep

    # Add the massive stationary particle at the center
    sim.add(m=M)  # Central mass

    # Add the small particle with random initial conditions
    initial_distance = np.random.uniform(0.0, unit_length)
    initial_angle = np.random.uniform(0, 2 * np.pi)
    x = initial_distance * np.cos(initial_angle)
    y = initial_distance * np.sin(initial_angle)

    velocity_angle = np.random.uniform(0, 2 * np.pi)  # Random velocity direction
    vx = velocity * np.cos(velocity_angle)
    vy = velocity * np.sin(velocity_angle)

    # Add the small particle as a test particle (mass=0)
    sim.add(x=x, y=y, vx=vx, vy=vy)  # Mass is 0 by default for test particles

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
            x, y = sim.particles[1].x, sim.particles[1].y
            vx, vy = sim.particles[1].vx, sim.particles[1].vy

            # Record the position and velocity every 10 time steps
            f.write(f"{sim.t:.6f}, {x:.6f}, {y:.6f}, {vx:.6f}, {vy:.6f}\n")

            # Check if the particle is inside the left or right triangular region
            if in_triangle(x, y, left=True):
                left_time_count += 1
            elif in_triangle(x, y, left=False):
                right_time_count += 1

    left_cone_probability = left_time_count / (total_time / sim.dt)
    right_cone_probability = right_time_count / (total_time / sim.dt)
    return left_cone_probability, right_cone_probability

# Running the simulation 100 times with test particles
n_simulations = 100
left_probabilities = []
right_probabilities = []

for simulation_number in range(1, n_simulations + 1):
    left_prob, right_prob = run_simulation(simulation_number)
    left_probabilities.append(left_prob)
    right_probabilities.append(right_prob)

# Calculate average probabilities
average_left_prob = np.mean(left_probabilities)
average_right_prob = np.mean(right_probabilities)

print(f"Average probability of the particle being in the left cone: {average_left_prob:.3f}")
print(f"Average probability of the particle being in the right cone: {average_right_prob:.3f}")
