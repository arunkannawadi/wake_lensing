from ._base_profile import MassProfile

__all__ = (
    "UniformDensityMassProfile",
)


class UniformDensityMassProfile(MassProfile):
    """Sphere of uniform density up to r_s.

    Parameters
    ----------
    r_s : `float`
        Radius containing all of the mass.
    """

    def __init__(self, *, r_s: float):
        self.r_s = r_s

    def __call__(self, r):
        return (r/self.r_s)**3
