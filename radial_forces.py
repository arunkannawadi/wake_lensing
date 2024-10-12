"""Collection of radial forces that can be used in rebound simulations."""

from typing import TYPE_CHECKING, Callable
import numpy as np

if TYPE_CHECKING:
    from .mass_profiles._base_profile import MassProfile


__all__ = (
    "RadialForce",
)


class RadialForce:
    """Class for radial forces."""

    G = 1
    """Gravitational constant."""

    def __init__(self, *, M):
        self.M = M
        self.mass_profile: MassProfile | None = None

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        if not value >= 0:
            raise ValueError("Total mass has to be a non-negative value.")
        self._M = value

    def __call__(self, reb_sim):
        if self.mass_profile is None:
            raise ValueError("Mass profile has to be set.")

        ps = reb_sim.contents.particles  # Access particles via reb_sim.contents
        for particle in ps[1:]:
            r = (particle.x**2 + particle.y**2 + particle.z**2)**0.5
            a_r = -4 * np.pi * self.G * self.M * self.mass_profile(r) / r**3

            particle.ax = a_r * particle.x
            particle.ay = a_r * particle.y
            particle.az = a_r * particle.z



