import numpy as np
import os

# Parameters
cone_angle_half = 30 * np.pi / 180  # Half of the opening angle in radians
output_folder = "simd"

# Precompute the cone direction and cosine of the cone angle
cone_direction_left = np.array([1, 0, 0])  # Along positive x-axis
cone_direction_right = -cone_direction_left  # Along negative x-axis
cos_cone_angle = np.cos(cone_angle_half)  # Cosine of the cone half-angle

# Function to check if points are inside a cone using dot products
def in_cone(positions, cone_direction):
    norms = np.linalg.norm(positions, axis=1)
    valid = norms > 0
    unit_vectors = np.zeros_like(positions)
    unit_vectors[valid] = positions[valid] / norms[valid][:, np.newaxis]
    cos_angles = unit_vectors @ cone_direction
    return cos_angles >= cos_cone_angle

# Function to calculate cone probabilities from the simulation data
def calculate_cone_probabilities(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None, None

    data = np.loadtxt(file_path, delimiter=',', skiprows=1)

    particle_ids = data[:, 0].astype(int)
    unique_particles = np.unique(particle_ids)
    total_timesteps = len(np.unique(data[:, 1]))  # Number of timesteps

    left_time_counts = []
    right_time_counts = []
    all_positions = []

    for particle_id in unique_particles:
        particle_data = data[particle_ids == particle_id]
        positions = particle_data[:, 2:5]  # x, y, z columns
        all_positions.append(positions)

        # Check if positions are inside the left cone
        left_in_cone = in_cone(positions, cone_direction_left)
        left_count = np.sum(left_in_cone)
        left_time_counts.append(left_count)

        # Check if positions are inside the right cone
        right_in_cone = in_cone(positions, cone_direction_right)
        right_count = np.sum(right_in_cone)
        right_time_counts.append(right_count)

    left_cone_probabilities = np.array(left_time_counts) / total_timesteps
    right_cone_probabilities = np.array(right_time_counts) / total_timesteps

    # Combine all particle positions into a single array for moment analysis
    all_positions = np.vstack(all_positions)

    return left_cone_probabilities, right_cone_probabilities, all_positions

# Function to calculate the mean position of particles (centroid)
def calculate_means(positions):
    return np.mean(positions, axis=0)  # Mean x, y, z for all particles

# Function to calculate the covariance matrix (second moments)
def calculate_covariance_matrix(positions):
    mean_position = calculate_means(positions)
    centered_positions = positions - mean_position
    covariance_matrix = np.cov(centered_positions, rowvar=False)
    return covariance_matrix

# Calculate probabilities and 3D moments from the output file
output_file = os.path.join(output_folder, "combined_simulation.txt")
left_probs, right_probs, all_positions = calculate_cone_probabilities(output_file)

# Calculate average probabilities
average_left_prob = np.mean(left_probs)
average_right_prob = np.mean(right_probs)

# Output probability results
print(f"Average probability of particles in left cone: {average_left_prob:.3f}")
print(f"Average probability of particles in right cone: {average_right_prob:.3f}")

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
