#!/usr/bin/env python3
import os
import numpy as np
import rebound
import matplotlib.pyplot as plt

def calculate_means(positions):
    """
    Calculate the centroid (mean position) of a set of positions.
    
    Parameters
    ----------
    positions : numpy.ndarray
        An (N,3) array of particle positions.
        
    Returns
    -------
    numpy.ndarray
        A length-3 array with the mean x, y, and z.
    """
    return np.mean(positions, axis=0)

def calculate_covariance_matrix(positions):
    """
    Calculate the covariance matrix (second moments) of a set of positions.
    
    Parameters
    ----------
    positions : numpy.ndarray
        An (N,3) array of particle positions.
        
    Returns
    -------
    numpy.ndarray
        The 3x3 covariance matrix.
    """
    # np.cov expects each row to be a variable (if rowvar=True) so we transpose.
    return np.cov(positions.T)


def in_cone(position, apex, direction, aperture_angle):
    """
    Check if a given position lies within a cone defined by its apex,
    central axis (direction), and full opening angle.
    
    Parameters
    ----------
    position : numpy.ndarray
        The (x,y,z) coordinates of the particle.
    apex : numpy.ndarray
        The (x,y,z) coordinates of the cone apex.
    direction : numpy.ndarray
        A unit vector pointing along the central axis of the cone.
    aperture_angle : float
        The full opening angle of the cone (in radians).

    Returns
    -------
    bool
        True if the particle lies within the cone, False otherwise.
    """
    half_angle = aperture_angle / 2.0
    vec = position - apex
    norm = np.linalg.norm(vec)
    # If the particle is exactly at the apex, count it as inside.
    if norm == 0:
        return True
    cos_angle = np.dot(vec, direction) / (norm * np.linalg.norm(direction))
    # Clip due to numerical issues.
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle < half_angle

def compute_com_velocity(particles):
    """Compute the center-of-mass velocity for all particles."""
    total_mass = sum(p.m for p in particles)
    vx = sum(p.m * p.vx for p in particles) / total_mass
    vy = sum(p.m * p.vy for p in particles) / total_mass
    vz = sum(p.m * p.vz for p in particles) / total_mass
    return np.array([vx, vy, vz])

def compute_angular_momentum(particles):
    """Compute the total angular momentum vector of the system."""
    L = np.array([0.0, 0.0, 0.0])
    for p in particles:
        r = np.array([p.x, p.y, p.z])
        v = np.array([p.vx, p.vy, p.vz])
        L += p.m * np.cross(r, v)
    return L

def compute_kinetic_energy(particles):
    """Compute the total kinetic energy of the system."""
    K = 0.0
    for p in particles:
        v2 = p.vx**2 + p.vy**2 + p.vz**2
        K += 0.5 * p.m * v2
    return K

def compute_potential_energy(particles, G):
    """Compute the total gravitational potential energy of the system.
    Note: This double-loop sums over each unique pair.
    """
    U = 0.0
    N = len(particles)
    for i in range(N):
        for j in range(i+1, N):
            dx = particles[i].x - particles[j].x
            dy = particles[i].y - particles[j].y
            dz = particles[i].z - particles[j].z
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            if r != 0:
                U += -G * particles[i].m * particles[j].m / r
    return U

# File produced by simulation code
archive_filename = "sim_nfw_one_blackhole.bin"

if not os.path.exists(archive_filename):
    raise FileNotFoundError(f"Archive file {archive_filename} not found.")

# Load the simulation archive.
archive = rebound.Simulationarchive(archive_filename)

# Pick the final snapshot for analysis.
sim_final = archive[-1]
print(f"Analyzing snapshot at simulation time t = {sim_final.t}")

# Exclude the central massive particle (assumed to be at index 0)
# and get positions for all test particles.
positions = np.array([[p.x, p.y, p.z] for p in sim_final.particles[1:]])
n_particles = len(positions)
print(f"Number of test particles (excluding the central object): {n_particles}")

# Calculate 3D moments for the final snapshot.
mean_position = calculate_means(positions)
covariance_matrix = calculate_covariance_matrix(positions)
variances = np.diag(covariance_matrix)

print("\n3D Distribution Analysis (Final Snapshot):")
print(f"Mean position (centroid): {mean_position}")
print("Covariance matrix:")
print(covariance_matrix)
print("Variances along x, y, z:", variances)

if np.allclose(variances, variances[0], rtol=0.1):
    print("\nThe distribution is approximately spherically symmetric.")
else:
    print("\nThe distribution deviates from spherical symmetry.")

# Cone probability tests for the final snapshot.
aperture_angle = 30 * np.pi / 180  # full opening angle in radians
apex = np.array([0.0, 0.0, 0.0])
# Left cone: central axis pointing in the negative x direction.
left_direction = np.array([-1.0, 0.0, 0.0])
left_direction /= np.linalg.norm(left_direction)
# Right cone: central axis pointing in the positive x direction.
right_direction = np.array([1.0, 0.0, 0.0])
right_direction /= np.linalg.norm(right_direction)

left_in_cone = [in_cone(pos, apex, left_direction, aperture_angle) for pos in positions]
right_in_cone = [in_cone(pos, apex, right_direction, aperture_angle) for pos in positions]

left_prob = np.mean(left_in_cone)
right_prob = np.mean(right_in_cone)

print(f"\nProbability of particles in the left cone: {left_prob:.3f}")
print(f"Probability of particles in the right cone: {right_prob:.3f}")

# --------------
# Moments Evolution Over Time
# --------------

# Prepare lists to store time series data.
times = []
centroid_list = []    # Each element is a 3-element array for (mean x, y, z)
variance_list = []    # Each element is a 3-element array for variances along x, y, z

com_velocity_list = []  # Center-of-mass velocity (x, y, z)
angular_momentum_list = []  # Angular momentum vector (Lx, Ly, Lz)

# Loop over all snapshots in the archive.
for sim in archive:
    times.append(sim.t)
    # Exclude the central object.
    pos = np.array([[p.x, p.y, p.z] for p in sim.particles[1:]])
    centroid_list.append(calculate_means(pos))
    variance_list.append(np.diag(calculate_covariance_matrix(pos)))
    
    particles = sim.particles  # All particles
    com_velocity_list.append(compute_com_velocity(particles))
    angular_momentum_list.append(compute_angular_momentum(particles))

# Convert lists to numpy arrays.
times = np.array(times)
centroids = np.array(centroid_list)      # shape: (num_snapshots, 3)
variances_array = np.array(variance_list)  # shape: (num_snapshots, 3)
com_velocities = np.array(com_velocity_list)  # shape: (num_snapshots, 3)
angular_momenta = np.array(angular_momentum_list)  # shape: (num_snapshots, 3)

# Plot the evolution of the centroid.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(times, centroids[:, 0], label='Mean x')
ax1.plot(times, centroids[:, 1], label='Mean y')
ax1.plot(times, centroids[:, 2], label='Mean z')
ax1.set_xlabel('Time')
ax1.set_ylabel('Centroid Position')
ax1.set_title('Evolution of the Centroid Over Time')
ax1.legend()
ax1.grid(True)

# Plot the evolution of the variances.
ax2.plot(times, variances_array[:, 0], label='Variance x')
ax2.plot(times, variances_array[:, 1], label='Variance y')
ax2.plot(times, variances_array[:, 2], label='Variance z')
ax2.set_xlabel('Time')
ax2.set_ylabel('Variance')
ax2.set_title('Evolution of Variances Over Time')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 1. Center-of-Mass Velocity Evolution
plt.figure(figsize=(8, 6))
plt.plot(times, com_velocities[:, 0], label='COM Velocity x')
plt.plot(times, com_velocities[:, 1], label='COM Velocity y')
plt.plot(times, com_velocities[:, 2], label='COM Velocity z')
plt.xlabel('Time')
plt.ylabel('COM Velocity (unit length / unit time)')
plt.title('Evolution of the Center-of-Mass Velocity')
plt.legend()
plt.grid(True)
plt.show()

# 2. Angular Momentum Evolution
plt.figure(figsize=(8, 6))
plt.plot(times, angular_momenta[:, 0], label='Angular Momentum Lx')
plt.plot(times, angular_momenta[:, 1], label='Angular Momentum Ly')
plt.plot(times, angular_momenta[:, 2], label='Angular Momentum Lz')
plt.xlabel('Time')
plt.ylabel('Angular Momentum (simulation units)')
plt.title('Angular Momentum Evolution Over Time')
plt.legend()
plt.grid(True)
plt.show()

# 3. Kinetic, Potential, and Total Energy Evolution
# Reload the archive to ensure we iterate over all snapshots again.
archive_energy = rebound.Simulationarchive(archive_filename)
kinetic_energies = []
potential_energies = []
total_energies = []
times_energy = []
for sim in archive_energy:
    times_energy.append(sim.t)
    ke = compute_kinetic_energy(sim.particles)
    pe = compute_potential_energy(sim.particles, sim.G)
    kinetic_energies.append(ke)
    potential_energies.append(pe)
    total_energies.append(ke + pe)

times_energy = np.array(times_energy)
kinetic_energies = np.array(kinetic_energies)
potential_energies = np.array(potential_energies)
total_energies = np.array(total_energies)

# Plot Kinetic Energy Evolution
plt.figure(figsize=(8, 6))
plt.plot(times_energy, kinetic_energies, label='Kinetic Energy')
plt.xlabel('Time')
plt.ylabel('Kinetic Energy')
plt.title('Kinetic Energy Evolution Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot Potential Energy Evolution
plt.figure(figsize=(8, 6))
plt.plot(times_energy, potential_energies, label='Potential Energy', color='orange')
plt.xlabel('Time')
plt.ylabel('Potential Energy')
plt.title('Potential Energy Evolution Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Energy Evolution
plt.figure(figsize=(8, 6))
plt.plot(times_energy, total_energies, label='Total Energy', color='green')
plt.xlabel('Time')
plt.ylabel('Total Energy')
plt.title('Total Energy Evolution Over Time')
plt.legend()
plt.grid(True)
plt.show()
