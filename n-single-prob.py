import numpy as np
import os
import sys
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
left_cone = Cone(apex=cone_apex_left, base=cone_base_left, aperture_angle=aperture_angle)
right_cone = Cone(apex=cone_apex_right, base=cone_base_right, aperture_angle=aperture_angle)

# File path for simulation data
output_file = os.path.join(output_folder, "combined_simulation.txt")

# Calculate probabilities and 3D moments from the output file
left_probs, right_probs, all_positions = ProbabilityCalculator.calculate(output_file, left_cone, right_cone)

# New list to store probabilities over time
left_probs_over_time = []
right_probs_over_time = []

for timestep_data in output_file:  # Use output_data from your simulation loop
    positions = timestep_data[:, 2:5]  # Extract x, y, z positions for each particle
    left_prob, right_prob = ProbabilityCalculator.calculate_probabilities(positions, left_cone, right_cone)
    left_probs_over_time.append(left_prob)
    right_probs_over_time.append(right_prob)

import matplotlib.pyplot as plt

# Convert lists to numpy arrays for easy indexing
left_probs_over_time = np.array(left_probs_over_time)
right_probs_over_time = np.array(right_probs_over_time)
time_steps = np.arange(len(left_probs_over_time)) * sim.dt  # Adjust for time based on timestep size

# Plot probabilities over time
plt.figure()
plt.plot(time_steps, left_probs_over_time, label='Left Cone Probability')
plt.plot(time_steps, right_probs_over_time, label='Right Cone Probability')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Probability Over Time in Left and Right Cones')
plt.legend()
plt.show()


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

