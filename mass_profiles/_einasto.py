from typing import Self
from scipy.special import gammaincc
from ._base_profile import MassProfile


__all__ = ["EinastoProfile"]


class EinastoProfile(MassProfile):
    """Family of Einasto density profiles characterized by two parameters.

    Parameters
    ----------
    r_s : `float`, optional
        Half-mass radius, i.e., the radius that encloses half the total mass.
    n : `float`, optional
        Einasto index, defining the steepness of the power law.

    Notes
    -----
    See Eq. 13 from https://www.aanda.org/articles/aa/full_html/2012/04/aa18543-11/aa18543-11.html
    """

    def __init__(self, *, r_s, n: float):
        self.r_s = r_s
        self.n = n

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value: float):
        if not value > 0:
            raise ValueError("Alpha has to be a positive value.")
        self._n = value

    def __call__(self: Self, r: float) -> float:
        s = 2.*self.n
        return gammaincc(3*self.n, s**1./self.n)
