#!/usr/bin/env python3
"""
Analysis script for the AMUSE output from your earlier script (traj.csv).

Expected input format (from your AMUSE writer):
t_Myr,i,x_kpc,y_kpc,z_kpc,vx_kms,vy_kms,vz_kms

What this does (analogous to your rebound/galpy analysis):
- Loads traj.csv
- Reconstructs per-snapshot arrays (positions/velocities for all particles)
- Computes centroid + covariance/variances of (x,y,z) at final time
- Computes left/right cone probabilities at final time
- Computes evolution over time of centroid and variances
- Computes COM velocity evolution (equal masses)
- Computes angular momentum evolution (equal masses)
- Computes "energy-like" evolution:
    * Kinetic energy: sum 1/2 v^2 (equal masses)
    * External potential energy: sum Phi_NFW(r) + Phi_BH(r)
      (NO pairwise self-gravity between tracers)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import G

import pandas as pd
from galpy.potential import NFWPotential, KeplerPotential, evaluatePotentials

# -----------------------------
# Utilities
# -----------------------------
def calculate_means(positions: np.ndarray) -> np.ndarray:
    return np.mean(positions, axis=0)

def calculate_covariance_matrix(positions: np.ndarray) -> np.ndarray:
    # positions shape (N,3)
    return np.cov(positions.T)

def in_cone(position, apex, direction, aperture_angle) -> bool:
    half_angle = aperture_angle / 2.0
    vec = position - apex
    norm = np.linalg.norm(vec)
    if norm == 0:
        return True
    direction = direction / np.linalg.norm(direction)
    cos_angle = np.dot(vec, direction) / norm
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle < half_angle

def compute_com_velocity(vxyz: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    if masses is None:
        return np.mean(vxyz, axis=0)
    mtot = np.sum(masses)
    return (masses[:, None] * vxyz).sum(axis=0) / mtot

def compute_angular_momentum(xyz: np.ndarray, vxyz: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    if masses is None:
        masses = np.ones(xyz.shape[0])
    return np.sum(masses[:, None] * np.cross(xyz, vxyz), axis=0)

def kinetic_energy(vxyz: np.ndarray, masses: np.ndarray | None = None) -> float:
    v2 = np.sum(vxyz**2, axis=1)
    if masses is None:
        masses = np.ones_like(v2)
    return float(0.5 * np.sum(masses * v2))

def potential_energy_external_galpy(xyz_kpc: np.ndarray, pot) -> float:
    """
    External potential energy proxy: sum Phi(x_i) in galpy units.

    We set ro=1 kpc, vo=1 km/s when building pot below, so:
      - inputs R,z are in kpc / ro = kpc
      - output Phi is in vo^2 = (km/s)^2
    Summing Phi gives something comparable over time as a conservation diagnostic.
    """
    x, y, z = xyz_kpc[:, 0], xyz_kpc[:, 1], xyz_kpc[:, 2]
    R = np.sqrt(x*x + y*y)  # kpc (since ro=1 kpc)
    zz = z                  # kpc
    phi = np.arctan2(y, x)
    Phi = evaluatePotentials(pot, R, zz, phi=phi)  # (km/s)^2 since vo=1 km/s
    return float(np.sum(Phi))

# -----------------------------
# Input file from AMUSE script
# -----------------------------
CSV_FILE = os.path.join("sim_nfw_amuse", "traj.csv")  # change if needed
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Could not find {CSV_FILE}. Update CSV_FILE path.")

df = pd.read_csv(CSV_FILE)

# Basic sanity
required_cols = ["t_Myr","i","x_kpc","y_kpc","z_kpc","vx_kms","vy_kms","vz_kms"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Sort to make grouping stable
df = df.sort_values(["t_Myr", "i"]).reset_index(drop=True)

times = np.sort(df["t_Myr"].unique())
n_times = len(times)
n_particles = df["i"].nunique()

print(f"Loaded {CSV_FILE}")
print(f"Snapshots: {n_times}, Particles: {n_particles}")
print(f"Final time: {times[-1]:.3f} Myr")

# -----------------------------
# Rebuild the SAME external potentials (for energy diagnostics)
# Match these to your simulation parameters
# -----------------------------
# From your AMUSE sim script:
CONC = 20.0
# halo_mass = (1e5/h) Msun -> convert to galpy mvir (in 1e12 Msun)
# If you want this exact, set HALO_MASS_MSUN = (1e5/h) yourself here.
# For analysis, it’s OK as long as it matches what you used in the run.
# ----
# Put the same halo_mass you used (numerically) here:
h = 0.7  # <-- EDIT if you want exact match; or hardcode the numeric halo_mass you used.
HALO_MASS_MSUN = 1e5 / h
mvir_1e12 = HALO_MASS_MSUN / 1e12

M_BH_MSUN = 4.3e6

# Choose galpy scaling so inputs are in kpc and outputs in (km/s)^2
ro_kpc = 1.0
vo_kms = 1.0

pot_halo = NFWPotential(mvir=mvir_1e12, conc=CONC, H=70.0, Om=0.3, overdens=200.0, ro=ro_kpc, vo=vo_kms)
pot_bh   = KeplerPotential(amp=(M_BH_MSUN * u.Msun), ro=ro_kpc, vo=vo_kms)
pot = [pot_halo, pot_bh]

# -----------------------------
# Final snapshot analysis
# -----------------------------
df_final = df[df["t_Myr"] == times[-1]].sort_values("i")

positions_final = df_final[["x_kpc","y_kpc","z_kpc"]].to_numpy(dtype=float)
v_final = df_final[["vx_kms","vy_kms","vz_kms"]].to_numpy(dtype=float)

mean_position = calculate_means(positions_final)
covariance_matrix = calculate_covariance_matrix(positions_final)
variances = np.diag(covariance_matrix)

print("\n3D Distribution Analysis (Final Snapshot):")
print(f"Mean position (centroid) [kpc]: {mean_position}")
print("Covariance matrix [kpc^2]:")
print(covariance_matrix)
print("Variances [kpc^2]:", variances)

# Cone tests (same as yours)
aperture_angle = 30.0 * np.pi / 180.0  # full opening angle
apex = np.array([0.0, 0.0, 0.0])
left_direction = np.array([-1.0, 0.0, 0.0])
right_direction = np.array([1.0, 0.0, 0.0])

left_prob = np.mean([in_cone(pos, apex, left_direction, aperture_angle) for pos in positions_final])
right_prob = np.mean([in_cone(pos, apex, right_direction, aperture_angle) for pos in positions_final])

print(f"\nProbability in left cone:  {left_prob:.3f}")
print(f"Probability in right cone: {right_prob:.3f}")

# -----------------------------
# Moments evolution over time
# -----------------------------
centroids = np.zeros((n_times, 3), dtype=float)
variances_array = np.zeros((n_times, 3), dtype=float)
com_velocities = np.zeros((n_times, 3), dtype=float)
angular_momenta = np.zeros((n_times, 3), dtype=float)

kinetic_energies = np.zeros(n_times, dtype=float)
potential_energies = np.zeros(n_times, dtype=float)
total_energies = np.zeros(n_times, dtype=float)

masses = None  # equal masses

for ti, t in enumerate(times):
    dft = df[df["t_Myr"] == t].sort_values("i")

    xyz = dft[["x_kpc","y_kpc","z_kpc"]].to_numpy(dtype=float)
    vxyz = dft[["vx_kms","vy_kms","vz_kms"]].to_numpy(dtype=float)

    centroids[ti] = calculate_means(xyz)
    variances_array[ti] = np.diag(calculate_covariance_matrix(xyz))

    com_velocities[ti] = compute_com_velocity(vxyz, masses=masses)
    angular_momenta[ti] = compute_angular_momentum(xyz, vxyz, masses=masses)

    kinetic_energies[ti] = kinetic_energy(vxyz, masses=masses)
    potential_energies[ti] = potential_energy_external_galpy(xyz, pot)
    total_energies[ti] = kinetic_energies[ti] + potential_energies[ti]

# -----------------------------
# Plots
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(times, centroids[:, 0], label="Mean x")
ax1.plot(times, centroids[:, 1], label="Mean y")
ax1.plot(times, centroids[:, 2], label="Mean z")
ax1.set_xlabel("Time [Myr]")
ax1.set_ylabel("Centroid [kpc]")
ax1.set_title("Centroid Evolution")
ax1.legend()
ax1.grid(True)

ax2.plot(times, variances_array[:, 0], label="Var x")
ax2.plot(times, variances_array[:, 1], label="Var y")
ax2.plot(times, variances_array[:, 2], label="Var z")
ax2.set_xlabel("Time [Myr]")
ax2.set_ylabel("Variance [kpc$^2$]")
ax2.set_title("Variance Evolution")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(times, com_velocities[:, 0], label="COM vx")
plt.plot(times, com_velocities[:, 1], label="COM vy")
plt.plot(times, com_velocities[:, 2], label="COM vz")
plt.xlabel("Time [Myr]")
plt.ylabel("COM velocity [km/s]")
plt.title("Center-of-Mass Velocity Evolution")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(times, angular_momenta[:, 0], label="Lx")
plt.plot(times, angular_momenta[:, 1], label="Ly")
plt.plot(times, angular_momenta[:, 2], label="Lz")
plt.xlabel("Time [Myr]")
plt.ylabel("Angular Momentum [kpc·km/s] (equal-mass units)")
plt.title("Angular Momentum Evolution")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(times, kinetic_energies, label="Kinetic")
plt.xlabel("Time [Myr]")
plt.ylabel("Kinetic (arb. units)")
plt.title("Kinetic Energy Evolution")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(times, potential_energies, label="External Potential")
plt.xlabel("Time [Myr]")
plt.ylabel("Potential [(km/s)$^2$ summed]")
plt.title("External Potential Energy Proxy (galpy)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(times, total_energies, label="Total (diagnostic)")
plt.xlabel("Time [Myr]")
plt.ylabel("Total (mixed proxy units)")
plt.title("Total Energy Proxy (Diagnostic)")
plt.legend()
plt.grid(True)
plt.show()
