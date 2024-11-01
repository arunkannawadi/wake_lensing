# shape_profiles/cylinder.py
import numpy as np
from ._base_shape import ShapeProfile

class Cylinder(ShapeProfile):
    """Class representing a cylinder shape."""

    def __init__(self, center: np.ndarray, radius: float, height: float):
        self.center = center
        self.radius = radius
        self.height = height

    def is_in_shape(self, point: np.ndarray) -> bool:
        # Calculate radial distance in the y-z plane
        radial_distance = np.sqrt((point[1] - self.center[1])**2 + (point[2] - self.center[2])**2)
        # Check height along the x-axis
        x_within_bounds = np.abs(point[0] - self.center[0]) <= self.height / 2
        return radial_distance <= self.radius and x_within_bounds
