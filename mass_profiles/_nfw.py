import numpy as np
from ._base_profile import MassProfile

__all__ = (
    "NFWProfile",
)


class NFWProfile(MassProfile):
    """NFW density profile.

    Parameters
    ----------
    r_s : `float`, optional
        Scale radius.
    """

    def __init__(self, *, r_s: float):
        self.r_s = r_s

    @property
    def r_s(self):
        return self._r_s

    @r_s.setter
    def r_s(self, value: float):
        if not value >= 0:
            raise ValueError("Scale radius has to be a non-negative value.")
        self._r_s = value

    def __call__(self, r) -> float:
        return np.log10(1. + r/self.r_s) - r/(r + self.r_s)
