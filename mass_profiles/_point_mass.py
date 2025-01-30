from ._base_profile import MassProfile

__all__ = ["PointMassProfile"]


class PointMassProfile(MassProfile):
    """Radial force model corresponding to a point source."""

    def __init__(self, *, r_s: float = 0.0):
        pass

    def __call__(self, r):
        return 1
