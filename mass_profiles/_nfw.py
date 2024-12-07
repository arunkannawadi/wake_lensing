import numpy as np
from . import MassProfile

__all__ = (
    "NFWProfile",
)


class NFWProfile(MassProfile):
    """NFW density profile.

    rho(r) = \rho_0/(r/r_s * (1 + r/r_s)^2)

    M(<r) = M * (np.log(1 + r/r_s) - r/(r + r_s))

    M := 4\pi\rho_0 r_s^3

    Parameters
    ----------
    r_s : `float`, optional
        Scale radius.
    """

    def __init__(self, *, r_s: float):
        self.r_s = r_s

    def __call__(self, r) -> float:
        return np.log10(1. + r/self.r_s) - r/(r + self.r_s)
