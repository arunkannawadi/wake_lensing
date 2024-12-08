# shape_profiles/cuboid.py
import numpy as np
from ._base_shape import ShapeProfile

class Cuboid(ShapeProfile):
    """Class representing a cuboid shape."""

    def __init__(self, center: np.ndarray, dimensions: np.ndarray):
        self.center = center
        self.dimensions = dimensions

    def is_in_shape(self, point: np.ndarray) -> bool:
        min_bound = self.center - self.dimensions / 2
        max_bound = self.center + self.dimensions / 2
        return np.all(point >= min_bound) and np.all(point <= max_bound)
