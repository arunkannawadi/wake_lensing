import os
import sys

import numpy as np

sys.path.append("/path/to/Physics")
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
output_file = os.path.join(output_folder, "combined_simulation_with_black_hole.txt")

# Calculate probabilities and 3D moments from the output file
left_probs, right_probs, all_positions = ProbabilityCalculator.calculate(
    output_file, left_cone, right_cone
)

# Calculate average probabilities
average_left_prob = np.mean(left_probs)
average_right_prob = np.mean(right_probs)

# Output probability results
print(f"Average probability of particles in left cone: {average_left_prob:.3f}")
print(f"Average probability of particles in right cone: {average_right_prob:.3f}")


# Function to calculate the mean position of particles (centroid)
def calculate_means(positions):
    return np.mean(positions, axis=0)  # Mean x, y, z for all particles


# Function to calculate the covariance matrix (second moments)
def calculate_covariance_matrix(positions):
    mean_position = calculate_means(positions)
    centered_positions = positions - mean_position
    covariance_matrix = np.cov(centered_positions, rowvar=False)
    return covariance_matrix


"""
# Calculate and output 3D moments
mean_position = calculate_means(all_positions)
covariance_matrix = calculate_covariance_matrix(all_positions)

print("\n3D Distribution Analysis:")
print(f"Mean position (centroid): {mean_position}")
print("Covariance matrix:")
print(covariance_matrix)

# Check for deviations from spherical symmetry
variances = np.diag(covariance_matrix)
print("\nVariances along x, y, z axes:", variances)
spherical_symmetry = np.allclose(variances, variances[0], rtol=0.1)

if spherical_symmetry:
    print("The distribution is approximately spherically symmetric.")
else:
    print("The distribution deviates from spherical symmetry.")
"""
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load the data
file_path = output_file  # Replace with your file path
data = pd.read_csv(file_path, sep="\t", names=["Particle", "Time step", "x", "y", "z"])
data = data.dropna(subset=["x", "y", "z"])

# Step 2: Group by time step
grouped = data.groupby("Time step")

# Initialize lists to store results
time_steps = []
moment_xy = []
moment_x2 = []
moment_y2 = []
anisotropy = []

# Step 3: Calculate moments for each time step
for time_step, group in grouped:
    x = group["x"].values
    y = group["y"].values

    # Compute mean and standard deviation
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)

    # Standardize the coordinates
    x_std = (x - mean_x) / std_x
    y_std = (y - mean_y) / std_y

    # Compute second moments using standardized coordinates
    x2 = np.mean(x_std**2)
    y2 = np.mean(y_std**2)
    xy = np.mean(x_std * y_std)

    # Anisotropy
    anisotropy_value = x2 - y2

    # Store results
    time_steps.append(time_step)
    moment_xy.append(xy)
    moment_x2.append(x2)
    moment_y2.append(y2)
    anisotropy.append(anisotropy_value)

# Step 4: Plot results
plt.figure(figsize=(12, 6))

# Plot xy moment
plt.subplot(1, 2, 1)
plt.plot(time_steps, moment_xy, label="⟨xy⟩")
plt.xlabel("Time step")
plt.ylabel("⟨xy⟩")
plt.title("Cross-Term Moment (⟨xy⟩)")
plt.legend()

# Plot anisotropy
plt.subplot(1, 2, 2)
plt.plot(time_steps, anisotropy, label="Anisotropy (xx-yy)")
plt.xlabel("Time step")
plt.ylabel("Anisotropy (⟨x²⟩ - ⟨y²⟩)")
plt.title("Anisotropy Over Time")
plt.legend()

plt.tight_layout()
plt.show()
