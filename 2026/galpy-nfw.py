import os
import numpy as np

import astropy.units as u
import halotools.sim_manager
import halotools.empirical_models

from amuse.lab import units, Particles, nbody_system
from amuse.community.ph4.interface import ph4
from galpy.potential import NFWPotential, to_amuse

np.random.seed(42)

# ----------------------------
# Parameters
# ----------------------------
n_particles = 100
output_folder = "sim_nfw_amuse_option_c"
os.makedirs(output_folder, exist_ok=True)

# Cosmology h
h = halotools.sim_manager.sim_defaults.cosmology.default_cosmology.get().h

# Halo + SMBH
halo_mass = (1e5 / h) * u.Msun
c = 20.0
M_bh = 4.3e6 * u.Msun  # SMBH mass (discrete particle)

# Integration horizon
t_end = 100.0 | units.Myr
dt = 0.01 | units.Myr
snap_dt = 1.0 | units.Myr

# Converter for ph4 (BH-only)
NBODY_MASS_UNIT = (M_bh.to_value(u.Msun) | units.MSun)
NBODY_LEN_UNIT = 1.0 | units.kpc
converter = nbody_system.nbody_to_si(NBODY_MASS_UNIT, NBODY_LEN_UNIT)

# ----------------------------
# 1) Sample ICs using halotools
# ----------------------------
table = halotools.empirical_models.NFWPhaseSpace().mc_generate_nfw_phase_space_points(
    Ngals=n_particles,
    mass=halo_mass.value,  # halotools expects Msun-ish convention
    conc=c,
)

# halotools: x,y,z in Mpc/h ; vx,vy,vz in km/s
x = (table["x"] * h * u.Mpc).to(u.kpc)
y = (table["y"] * h * u.Mpc).to(u.kpc)
z = (table["z"] * h * u.Mpc).to(u.kpc)
vx = (table["vx"] * (u.km / u.s))
vy = (table["vy"] * (u.km / u.s))
vz = (table["vz"] * (u.km / u.s))

# ----------------------------
# 2) Build halo-only galpy potential (Option C)
#    - Halo is a smooth external field
#    - SMBH is a discrete particle in ph4
# ----------------------------
mvir_1e12 = (halo_mass / (1e12 * u.Msun)).value
halo_pot = NFWPotential(conc=c, mvir=mvir_1e12, H=70.0, Om=0.3, overdens=200.0)

# Convert halo potential to an AMUSE "field" code
field_halo = to_amuse(halo_pot)

# ----------------------------
# 3) Make tracer particles (NOT added to ph4)
# ----------------------------
tracers = Particles(n_particles)
tracers.mass = 0.0 | units.MSun  # truly collisionless tracers (not used anywhere)

tracers.x = x.value | units.kpc
tracers.y = y.value | units.kpc
tracers.z = z.value | units.kpc

tracers.vx = vx.value | (units.km / units.s)
tracers.vy = vy.value | (units.km / units.s)
tracers.vz = vz.value | (units.km / units.s)

# ----------------------------
# 4) Make BH-only gravity code (discrete SMBH)
# ----------------------------
bh = Particles(1)
bh.mass = (M_bh.to_value(u.Msun) | units.MSun)
bh.x = 0 | units.kpc
bh.y = 0 | units.kpc
bh.z = 0 | units.kpc
bh.vx = 0 | (units.km / units.s)
bh.vy = 0 | (units.km / units.s)
bh.vz = 0 | (units.km / units.s)

bhgrav = ph4(converter)
bhgrav.particles.add_particles(bh)

# ----------------------------
# 5) Acceleration helper: a = a_halo + a_BH
# ----------------------------
def get_gravity(code, eps, x, y, z):
    """
    Wrapper to be robust to slightly different AMUSE field interfaces.
    Returns ax, ay, az (each with acceleration units).
    """
    if hasattr(code, "get_gravity_at_point"):
        return code.get_gravity_at_point(eps, x, y, z)
    raise AttributeError("Code does not expose get_gravity_at_point; cannot evaluate acceleration.")


def accel_tracers(trs):
    eps = 0 | units.kpc  # no softening
    ax_h, ay_h, az_h = get_gravity(field_halo, eps, trs.x, trs.y, trs.z)
    ax_b, ay_b, az_b = get_gravity(bhgrav, eps, trs.x, trs.y, trs.z)
    return ax_h + ax_b, ay_h + ay_b, az_h + az_b


# ----------------------------
# 6) Leapfrog integrator for tracers (BH fixed, halo fixed)
# ----------------------------
def leapfrog_step(trs, dt):
    # kick half
    ax, ay, az = accel_tracers(trs)
    trs.vx += 0.5 * dt * ax
    trs.vy += 0.5 * dt * ay
    trs.vz += 0.5 * dt * az

    # drift
    trs.x += dt * trs.vx
    trs.y += dt * trs.vy
    trs.z += dt * trs.vz

    # kick half
    ax, ay, az = accel_tracers(trs)
    trs.vx += 0.5 * dt * ax
    trs.vy += 0.5 * dt * ay
    trs.vz += 0.5 * dt * az


# ----------------------------
# 7) Evolve + dump snapshots
# ----------------------------
t = 0.0 | units.Myr
next_snap = 0.0 | units.Myr

out_path = os.path.join(output_folder, "traj.csv")
with open(out_path, "w") as f:
    f.write("t_Myr,i,x_kpc,y_kpc,z_kpc,vx_kms,vy_kms,vz_kms\n")

    while t < t_end:
        # advance one small step
        leapfrog_step(tracers, dt)
        t += dt

        # snapshot
        if t >= next_snap:
            for i, p in enumerate(tracers):
                f.write(
                    f"{t.value_in(units.Myr):.6f},{i},"
                    f"{p.x.value_in(units.kpc):.6f},{p.y.value_in(units.kpc):.6f},{p.z.value_in(units.kpc):.6f},"
                    f"{p.vx.value_in(units.km/units.s):.6f},{p.vy.value_in(units.km/units.s):.6f},{p.vz.value_in(units.km/units.s):.6f}\n"
                )
            next_snap += snap_dt

# cleanup
field_halo.stop()
bhgrav.stop()

print(f"Wrote {out_path}")
