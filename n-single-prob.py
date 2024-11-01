import numpy as np
import os

# Parameters
aperture_angle = 30 * np.pi / 180  # Full cone opening angle in radians
half_ap_cos = np.cos(aperture_angle / 2)
output_folder = "simd"

# Function to check if a point X lies inside the cone
def is_in_cone(x, apex, base, aperture_cos):
    # Vector from apex to point X
    apex_to_x = x - apex
    
    # Vector from apex to base of the cone
    axis_vector = base - apex
    
    # Check if X is in the infinite cone
    dot_product = np.dot(apex_to_x, axis_vector)
    is_in_infinite_cone = dot_product / (np.linalg.norm(apex_to_x) * np.linalg.norm(axis_vector)) > aperture_cos

    if not is_in_infinite_cone:
        return False
    
    # Check if X is below the cone's round cap
    projection_length = dot_product / np.linalg.norm(axis_vector)
    return projection_length < np.linalg.norm(axis_vector)

# Updated function to calculate cone probabilities from the simulation data
def calculate_cone_probabilities(file_path, aperture_cos, cone_apex_left, cone_base_left, cone_apex_right, cone_base_right):
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
        left_in_cone = np.array([is_in_cone(pos, cone_apex_left, cone_base_left, aperture_cos) for pos in positions])
        left_count = np.sum(left_in_cone)
        left_time_counts.append(left_count)

        # Check if positions are inside the right cone
        right_in_cone = np.array([is_in_cone(pos, cone_apex_right, cone_base_right, aperture_cos) for pos in positions])
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

# Define cone parameters
cone_apex_left = np.array([0, 0, 0])
cone_base_left = np.array([-1, 0, 0])  # -1 unit along the x-axis for the base center
cone_apex_right = cone_apex_left
cone_base_right = np.array([1, 0, 0])  # 1 unit along the x-axis for the base center

# Calculate probabilities and 3D moments from the output file
output_file = os.path.join(output_folder, "combined_simulation.txt")
left_probs, right_probs, all_positions = calculate_cone_probabilities(output_file, half_ap_cos, cone_apex_left, cone_base_left, cone_apex_right, cone_base_right)

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
