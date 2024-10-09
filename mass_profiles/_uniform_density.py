from ._base_profile import MassProfile

__all__ = (
    "UniformDensityMassProfile",
)


class UniformDensityMassProfile(MassProfile):
    """Sphere of uniform density up to r_h.

    Parameters
    ----------
    r_h : `float`, optional
        Halo size.
    """

    def __init__(self, *, r_h=1.0):
        self.r_h = r_h

    @property
    def r_h(self):
        return self._r_h

    @r_h.setter
    def r_h(self, value: float):
        if not value >= 0:
            raise ValueError("Halo size has to be a non-negative value.")
        self._r_h = value

    def __call__(self, r):
        return (r/self.r_h)**3
