import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Parameters
theta = 30 * np.pi / 180  # 30 degrees in radians
unit_length = 1.0
output_folder = "simd"  # Updated output folder


# Load the simulation data from a specific file
def load_simulation_data(file_path):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    particle_ids = data[:, 0]
    time_steps = data[:, 1]
    x_positions = data[:, 2]
    y_positions = data[:, 3]
    return particle_ids, time_steps, x_positions, y_positions


# Create a figure and axis for the animation
fig, ax = plt.subplots()
lines = []

# Set up the plot limits and other properties
ax.set_xlim(-unit_length, unit_length)
ax.set_ylim(-unit_length, unit_length)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

# Plot the central particle as a marker
central_particle = plt.Circle((0, 0), 0.05, color="orange")
ax.add_patch(central_particle)

# Draw the left triangle (hourglass shape)
ax.plot(
    [0, -unit_length], [0, -unit_length * np.tan(theta)], color="blue", linestyle="--"
)
ax.plot(
    [0, -unit_length], [0, unit_length * np.tan(theta)], color="blue", linestyle="--"
)

# Draw the right triangle (hourglass shape)
ax.plot(
    [0, unit_length], [0, -unit_length * np.tan(theta)], color="red", linestyle="--"
)
ax.plot([0, unit_length], [0, unit_length * np.tan(theta)], color="red", linestyle="--")


# Initialize the plot for animation
def init():
    for line in lines:
        line.set_data([], [])
    return lines


# Update function for animation
def update(frame):
    frame = frame * 2  # Skip frames to make particles move faster
    for line, (x_positions, y_positions) in zip(lines, all_positions):
        line.set_data(x_positions[:frame], y_positions[:frame])
    return lines


# Collect data from the simulation file
all_positions = []
simulation_file = "simd/cs1.txt"

# Load data for all particles from the file
particle_ids, time_steps, x_positions, y_positions = load_simulation_data(
    simulation_file
)

# Create a line object for each unique particle
unique_particle_ids = np.unique(particle_ids)[:50]
for particle_id in unique_particle_ids:
    particle_mask = particle_ids == particle_id
    x_pos = x_positions[particle_mask]
    y_pos = y_positions[particle_mask]

    (line,) = ax.plot([], [], lw=2)
    lines.append(line)
    all_positions.append((x_pos, y_pos))

# Create the animation
max_frames = len(np.unique(time_steps))  # Number of unique time steps
ani = animation.FuncAnimation(
    fig, update, frames=max_frames, init_func=init, blit=True, interval=0.01
)

# Display the animation
plt.show()
