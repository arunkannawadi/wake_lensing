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
output_file = os.path.join(output_folder, "combined_simulation_with_black_hole.txt")

# Calculate probabilities and 3D moments from the output file
left_probs, right_probs, all_positions = ProbabilityCalculator.calculate(output_file, left_cone, right_cone)

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
