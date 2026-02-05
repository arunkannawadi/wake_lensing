import os

import matplotlib.pyplot as plt
import numpy as np
from probability_profiles import Cone

from calculate_probabilities import ProbabilityCalculator

# Parameters
aperture_angle = 30 * np.pi / 180  # Full cone opening angle in radians
output_folder = "simd"

# Define cone parameters
cone_apex_left = np.array([0, 0, 0])
cone_base_left = np.array([-1, 0, 0])  # -1 unit along the x-axis for the base center
cone_apex_right = cone_apex_left
cone_base_right = np.array([1, 0, 0])  # 1 unit along the x-axis for the base center

# Create Cone objects for the left and right cones
left_cone = Cone(
    apex=cone_apex_left, base=cone_base_left, aperture_angle=aperture_angle
)
right_cone = Cone(
    apex=cone_apex_right, base=cone_base_right, aperture_angle=aperture_angle
)

# File path for simulation data
output_file = os.path.join(output_folder, "combined_simulation.txt")

# Load data with error handling
try:
    data = np.loadtxt(output_file, delimiter=",", skiprows=1)
except Exception as e:
    print(f"Error loading data from file: {e}")
    exit()

# Ensure the data is not empty
if data.size == 0:
    print("Error: The data file is empty or improperly formatted.")
    exit()

# Extract unique time steps and initialize counts
time_steps = np.unique(data[:, 1])  # Unique time steps
left_counts = []
right_counts = []

# Initialize lists for moments
moment_xy = []
moment_x2 = []
moment_y2 = []
anisotropy = []

# Loop through each time step
for t in time_steps:
    # Select particles at this time step
    time_data = data[data[:, 1] == t]
    if time_data.size == 0:
        left_counts.append(0)
        right_counts.append(0)
        moment_xy.append(0)
        moment_x2.append(0)
        moment_y2.append(0)
        anisotropy.append(0)
        continue

    positions = time_data[:, 2:5]  # x, y, z columns

    # Check membership in left and right cones
    left_count = sum(left_cone.is_in_shape(pos) for pos in positions)
    right_count = sum(right_cone.is_in_shape(pos) for pos in positions)
    left_counts.append(left_count)
    right_counts.append(right_count)

    # Extract x and y components for moment calculations
    x = time_data[:, 2]
    y = time_data[:, 3]

    # Calculate moments
    mean_xy = np.mean(x * y)
    mean_x2 = np.mean(x**2)
    mean_y2 = np.mean(y**2)
    anisotropy_value = mean_x2 - mean_y2

    # Store moments
    moment_xy.append(mean_xy)
    moment_x2.append(mean_x2)
    moment_y2.append(mean_y2)
    anisotropy.append(anisotropy_value)

# Plot the number of particles in each cone over time
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(time_steps, left_counts, label="Left Cone")
plt.plot(time_steps, right_counts, label="Right Cone")
plt.xlabel("Time Step")
plt.ylabel("Number of Particles in Cone")
plt.title("Number of Particles in Left and Right Cones Over Time")
plt.legend()

# Plot moments over time
plt.subplot(1, 2, 2)
plt.plot(time_steps, moment_xy, label="⟨xy⟩")
plt.plot(time_steps, moment_x2, label="⟨x²⟩")
plt.plot(time_steps, moment_y2, label="⟨y²⟩")
plt.plot(time_steps, anisotropy, label="Anisotropy (⟨x²⟩ - ⟨y²⟩)")
plt.xlabel("Time Step")
plt.ylabel("Moment Value")
plt.title("Moments of the Particle System Over Time")
plt.legend()

plt.tight_layout()
plt.show()
