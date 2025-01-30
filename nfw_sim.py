import logging
import os
import time  # Import the time module

import astropy.units as u
import halotools.empirical_models
import numpy as np
import rebound

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mass_profiles import *
from mass_profiles import NFWProfile
from radial_forces import (
    RadialForce,
)  # Import the RadialForce class from the radial_forces module

# Parameters
theta = 30 * np.pi / 180  # Half of the total opening angle (in radians)
velocity = 1.0  # Unit velocity
n_periods = 3  # Number of periods to simulate
unit_length = 1.0  # Define the region of observation as unit length
output_folder = "sim_nfw"  # Folder to store the output files

np.random.seed(42)

# Gravitational constant and central mass
M = 1.0  # Central mass
Msol = None
h = (
    halotools.sim_manager.sim_defaults.cosmology.default_cosmology.get().h
)  # Reduced Hubble constant
halo_mass = (1e5 / h) * u.Msun  # Mass of the dark matter halo in solar masses/h.

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# Function to calculate orbital period based on initial distance
def calculate_orbital_period(distance, M, G):
    return 2 * np.pi * np.sqrt(distance**3 / (G * M))


# Function to run a simulation with multiple test particles
def run_simulation_with_particles(
    n_particles, black_hole_distance=None, black_hole_approaching=False
):
    sim = rebound.Simulation()
    sim.units = (
        "pc",
        "Myr",
        "Msun",
    )
    G = sim.G
    sim.start_server(port=1234)
    sim.integrator = "whfast"
    sim.dt = 1e-1  # Small timestep
    sim.ri_ias15.min_dt = 0  # 2e-16  # Minimum timestep

    # Add an SMBH at the center of the halo.
    sim.add(m=(4.3 * 1e6 * u.Msun).to_value(sim.units["mass"].capitalize()))
    c = 20.0  # Concentration parameter
    # Add the massive stationary particle at the center
    # sim.add(m=M)  # Central mass

    # Add multiple small particles with random initial conditions
    orbital_periods = []
    table = (
        halotools.empirical_models.NFWPhaseSpace().mc_generate_nfw_phase_space_points(
            Ngals=n_particles,
            mass=halo_mass.value,
            conc=c,
        )
    )
    for i in range(n_particles):
        initial_distance = np.random.uniform(0.0, unit_length)
        theta = np.random.uniform(0, 2 * np.pi)  # Polar angle
        phi = np.random.uniform(0, np.pi)  # Azimuthal angle
        x = initial_distance * np.sin(phi) * np.cos(theta)
        y = initial_distance * np.sin(phi) * np.sin(theta)
        z = initial_distance * np.cos(phi)
        vx = 0.0
        vy = 0.0
        vz = 0.0

        # Halotools returns a table with the following columns:
        # Positions are in units Mpc/h.
        # Velocities are in units km/s.
        x, y, z, vx, vy, vz = table[i][:6]

        # # HACK
        # y, z = 0.0, 0.0
        # x = table['radial_position'][i]
        # vx, vz = 0.0, 0.0
        # vy = table['radial_velocity'][i]

        x *= h * u.Mpc
        y *= h * u.Mpc
        z *= h * u.Mpc
        vx *= u.km / u.s
        vy *= u.km / u.s
        vz *= u.km / u.s

        sim.add(
            x=x.to_value(sim.units["length"]),
            y=y.to_value(sim.units["length"]),
            z=z.to_value(sim.units["length"]),
            vx=vx.to_value(
                getattr(u, sim.units["length"])
                / getattr(u, sim.units["time"].capitalize())
            ),
            vy=vy.to_value(
                getattr(u, sim.units["length"])
                / getattr(u, sim.units["time"].capitalize())
            ),
            vz=vz.to_value(
                getattr(u, sim.units["length"])
                / getattr(u, sim.units["time"].capitalize())
            ),
        )

        # Calculate the orbital period for this particle
        orbital_period = calculate_orbital_period(initial_distance, M, sim.G)
        orbital_periods.append(orbital_period)

        logger.debug(f"Particle {i}: x={x}, y={y}, z={z}, vx={vx}, vy={vy}, vz={vz}")

    # Set N_active to ensure only the central mass and black hole (if added) influence the test particles
    sim.N_active = sim.N

    # Add the additional force to the simulation
    our_mass = halo_mass.to_value(sim.units["mass"].capitalize()) * h
    our_mass /= np.log(1.0 + c) - c / (1.0 + c)

    radial_force = RadialForce(M=our_mass)
    radial_force.G = sim.G
    r_s = 1.0 * u.kpc  # Scale radius of the Milky Way NFW profile.
    radial_force.mass_profile = NFWProfile(r_s=r_s.to_value(sim.units["length"]))
    sim.additional_forces = radial_force

    # Define the total simulation time based on the longest orbital period
    # total_time = 100000 # n_periods * max(orbital_periods)  # Total time to simulate
    total_time = 10000  # 50 * max(sim.particles[pn].P for pn in range(1, sim.N-1))
    logger.info("Total simulation time: %f", total_time)

    # Store the data for all particles and timesteps
    output_data = []

    sim.save_to_file("sim_nfw.bin", interval=0.1)
    sim.integrate(total_time)

    # writeout_interval = 10.
    # while sim.t < total_time:
    #     sim.integrate(sim.t + writeout_interval)  # Integrate simulation
    #     for i in range(1, sim.N):
    #         output_data.append([i, sim.t, sim.particles[i].x, sim.particles[i].y, sim.particles[i].z])

    # Convert output_data to a NumPy array
    output_data = np.array(output_data)

    # Define the output file for this simulation
    output_file = os.path.join(output_folder, f"cs1.txt")

    # Use numpy.savetxt to write all data at once
    header = "Particle, Time step, x, y, z"
    np.savetxt(output_file, output_data, fmt="%.6f", delimiter=",", header=header)


# Start the timer
start_time = time.time()

# Running the simulation with 100 particles
n_particles = 5**3

run_simulation_with_particles(n_particles, black_hole_distance=None)
print("Running simulation with black hole infinitely far")

# End the timer
end_time = time.time()

# Calculate runtime
runtime = end_time - start_time
print(f"Simulation with {n_particles} particles completed in {runtime:.2f} seconds.")
