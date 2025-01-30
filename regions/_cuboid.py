# shape_profiles/cuboid.py
import numpy as np

from ._region import Region

__all__ = (
    "Cube",
    "Cuboid",
)


class Cuboid(Region):
    """Class representing a cuboid shape."""

    def __init__(self, *, dimensions: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.dimensions = dimensions

    def contains(self, point: np.ndarray) -> bool:
        # Docstring inherited.
        min_bound = self.center - self.dimensions / 2
        max_bound = self.center + self.dimensions / 2
        return np.all(point >= min_bound) and np.all(point <= max_bound)


class Cube(Cuboid):
    """Class representing a cube shape."""

    def __init__(self, side_length: float, **kwargs):
        super().__init__(dimensions=None, **kwargs)
        self.side_length = side_length
        self._dimensions = np.full(3, side_length)

    @property
    def dimensions(self):
        return np.full(3, self.side_length)

    @dimensions.setter
    def dimensions(self, value: None):
        if value is not None:
            raise ValueError("Cannot set dimensions on a Cube instance.")
