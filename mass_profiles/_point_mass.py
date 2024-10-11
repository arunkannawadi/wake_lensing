from ._base_profile import MassProfile

__all__ = ["PointMassProfile"]


class PointMassProfile(MassProfile):
    """Radial force model corresponding to a point source."""

    def __call__(self, r):
        return 1
