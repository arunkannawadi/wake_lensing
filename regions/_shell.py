import numpy as np

from ._base_shape import Region

__all__ = (
    "Shell",
    "Sphere",
)


class Shell(Region):
    """Class representing a shell region."""

    def __init__(self, r_inner: float, r_outer: float, **kwargs):
        super().__init__(**kwargs)
        self.r_inner = r_inner
        self.r_outer = r_outer

    def contains(self, point: np.ndarray) -> bool:
        r = np.linalg.norm(point - self.center)
        return self.r_inner <= r <= self.r_outer


class Sphere(Shell):
    """Class representing a sphere region."""

    def __init__(self, radius: float, **kwargs):
        super().__init__(r_inner=None, r_outer=None, **kwargs)
        self.radius = radius

    @property
    def r_inner(self):
        return 0.0

    @property
    def r_outer(self):
        return self.radius

    @r_inner.setter
    def r_inner(self, value: None = None):
        if value is not None:
            raise ValueError("Cannot set r_inner on a Sphere instance.")

    @r_outer.setter
    def r_outer(self, value: None = None):
        if value is not None:
            raise ValueError("Cannot set r_outer on a Sphere instance.")
